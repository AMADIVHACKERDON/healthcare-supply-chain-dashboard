import requests

def get_data(url, params=None, headers=None):
    """
    Makes a GET request to the specified URL.

    Args:
        url (str): The API endpoint URL.
        params (dict, optional): Query parameters to send with the request.
        headers (dict, optional): HTTP headers to send with the request.

    Returns:
        dict: Parsed JSON response from the API.
    """
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

# Example usage
url = "https://api.example.com/data"
params = {"key": "value"}
headers = {"Authorization": "Bearer YOUR_TOKEN"}
response = get_data(url, params, headers)
print(response)








import requests

class APIClient:
    """
    A client to interact with an API.
    """
    def __init__(self, base_url, token=None):
        """
        Initializes the API client.

        Args:
            base_url (str): The base URL of the API.
            token (str, optional): The API token for authentication.
        """
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}" if token else None,
            "Content-Type": "application/json"
        }

    def make_request(self, endpoint, method="GET", params=None, data=None):
        """
        Makes a request to the API.

        Args:
            endpoint (str): The API endpoint (appended to base_url).
            method (str): HTTP method ("GET", "POST", etc.).
            params (dict, optional): Query parameters.
            data (dict, optional): Request payload for POST/PUT.

        Returns:
            dict: Parsed JSON response from the API.
        """
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.request(
                method, url, headers=self.headers, params=params, json=data
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None

# Example usage
client = APIClient("https://api.example.com", token="YOUR_TOKEN")
response = client.make_request("data", method="GET", params={"key": "value"})
print(response)
