import os
import unittest

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


class TestGoogleApi(unittest.TestCase):
    _fallback_models = (
        "Gemini 2.5 Flash",
        "Gemini 2.5 Flash-Lite"
    )

    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv()
        cls.api_key = os.getenv("GOOGLE_API_KEY")
        if not cls.api_key:
            raise unittest.SkipTest("GOOGLE_API_KEY is not set")

    def _llm_kwargs(self):
        kwargs = {}
        api_version = os.getenv("GOOGLE_API_VERSION")
        if api_version:
            kwargs["client_options"] = {"api_version": api_version}

        use_vertex = os.getenv("GOOGLE_VERTEXAI") or os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
        if use_vertex and use_vertex.lower() in ("1", "true", "yes", "on"):
            kwargs["vertexai"] = True
            kwargs["project"] = os.getenv("GOOGLE_CLOUD_PROJECT")
            kwargs["location"] = os.getenv("GOOGLE_CLOUD_LOCATION")

        return kwargs

    def _api_info(self, model: str) -> str:
        api_version = os.getenv("GOOGLE_API_VERSION") or "v1beta"
        use_vertex = os.getenv("GOOGLE_VERTEXAI") or os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
        use_vertex = use_vertex and use_vertex.lower() in ("1", "true", "yes", "on")
        project = os.getenv("GOOGLE_CLOUD_PROJECT") if use_vertex else None
        location = os.getenv("GOOGLE_CLOUD_LOCATION") if use_vertex else None
        return (
            "API=google-genai "
            f"api_version={api_version} "
            f"vertexai={bool(use_vertex)} "
            f"project={project or '-'} "
            f"location={location or '-'} "
            f"model={model}"
        )

    def _normalize_model_names(self, name: str):
        names = []
        if not name:
            return names

        lowered = name.strip().lower()
        if " " in lowered:
            names.append(lowered.replace(" ", "-"))
        names.append(name)
        return list(dict.fromkeys(n for n in names if n))

    def _list_generate_content_models(self):
        try:
            from google import genai
        except Exception as exc:
            return [], f"google-genai import failed: {exc}"

        try:
            api_version = os.getenv("GOOGLE_API_VERSION")
            http_options = {"api_version": api_version} if api_version else None
            use_vertex = os.getenv("GOOGLE_VERTEXAI") or os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
            if use_vertex and use_vertex.lower() in ("1", "true", "yes", "on"):
                client = genai.Client(
                    vertexai=True,
                    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                    location=os.getenv("GOOGLE_CLOUD_LOCATION"),
                    http_options=http_options,
                )
            else:
                client = genai.Client(api_key=self.api_key, http_options=http_options)

            models = []
            for model in client.models.list():
                name = getattr(model, "name", None)
                methods = getattr(model, "supported_generation_methods", []) or []
                if name and "generateContent" in methods:
                    models.append(name)
            return models, None
        except Exception as exc:
            return [], str(exc)

    def test_chat_completion(self) -> None:
        model_override = os.getenv("GOOGLE_MODEL")
        candidates = []
        if model_override:
            for candidate in self._normalize_model_names(model_override):
                candidates.append(candidate)
        candidates.extend(self._fallback_models)

        last_not_found = None
        for model in candidates:
            print(self._api_info(model))
            llm = ChatGoogleGenerativeAI(model=model, temperature=0, **self._llm_kwargs())
            try:
                response = llm.invoke("你现在是什么模型")
            except Exception as exc:
                msg = str(exc)
                if "INVALID_ARGUMENT" in msg and "model name format" in msg:
                    last_not_found = msg
                    continue
                if "NOT_FOUND" in msg or "not found" in msg or "not supported" in msg:
                    last_not_found = msg
                    continue
                self.fail(f"Google API call failed for model '{model}': {exc}")
                return

            content = (response.content or "").strip()
            print(response)
            # self.assertTrue(content, f"Empty response from '{model}'")
            return

        discovered_models, list_error = self._list_generate_content_models()
        if discovered_models:
            normalized = []
            for name in discovered_models:
                normalized.append(name)
                if name.startswith("models/"):
                    normalized.append(name.replace("models/", "", 1))
            for name in normalized[:4]:
                print(self._api_info(name))
                llm = ChatGoogleGenerativeAI(model=name, temperature=0, **self._llm_kwargs())
                try:
                    response = llm.invoke("Reply with exactly: OK")
                except Exception as exc:
                    msg = str(exc)
                    if "INVALID_ARGUMENT" in msg and "model name format" in msg:
                        last_not_found = msg
                        continue
                    if "NOT_FOUND" in msg or "not found" in msg or "not supported" in msg:
                        last_not_found = msg
                        continue
                    self.fail(f"Google API call failed for model '{name}': {exc}")
                    return

                content = (response.content or "").strip()
                self.assertTrue(content, f"Empty response from '{name}'")
                return

        self.fail(
            "No compatible model found. Set GOOGLE_MODEL to a valid model name. "
            f"Last error: {last_not_found}. Available models: {discovered_models}. "
            f"List error: {list_error}. "
            "If you are using Vertex AI, set GOOGLE_VERTEXAI=true and configure "
            "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION."
        )


if __name__ == "__main__":
    unittest.main()
