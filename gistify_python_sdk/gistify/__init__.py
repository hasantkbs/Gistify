import requests

class GistifyClient:
    def __init__(self, base_url="https://api.gistify.com"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}

    def login(self, email, password):
        """
        Logs in a user and saves the access token.
        """
        response = requests.post(f"{self.base_url}/auth/login", data={"username": email, "password": password})
        response.raise_for_status()
        self.headers["Authorization"] = f"Bearer {response.json()['access_token']}"

    def summarize_text(self, text, length="medium", summary_type="abstractive", extractive_method="textrank", tone="neutral"):
        """
        Summarizes a piece of text.
        """
        response = requests.post(
            f"{self.base_url}/summarize",
            headers=self.headers,
            json={
                "text": text,
                "summary_length": length,
                "summary_type": summary_type,
                "extractive_method": extractive_method,
                "tone": tone,
            },
        )
        response.raise_for_status()
        return response.json()

    def summarize_file(self, file_path, length="medium", summary_type="abstractive", extractive_method="textrank", tone="neutral"):
        """
        Summarizes a file.
        """
        with open(file_path, "rb") as f:
            response = requests.post(
                f"{self.base_url}/summarize_file",
                headers=self.headers,
                files={"file": f},
                data={
                    "summary_length": length,
                    "summary_type": summary_type,
                    "extractive_method": extractive_method,
                    "tone": tone,
                },
            )
        response.raise_for_status()
        return response.json()

    def summarize_url(self, url, length="medium", summary_type="abstractive", extractive_method="textrank", tone="neutral"):
        """
        Summarizes a URL.
        """
        response = requests.post(
            f"{self.base_url}/summarize_url",
            headers=self.headers,
            json={
                "url": url,
                "summary_length": length,
                "summary_type": summary_type,
                "extractive_method": extractive_method,
                "tone": tone,
            },
        )
        response.raise_for_status()
        return response.json()

    def get_task_status(self, task_id):
        """
        Gets the status of a task.
        """
        response = requests.get(f"{self.base_url}/tasks/{task_id}", headers=self.headers)
        response.raise_for_status()
        return response.json()
