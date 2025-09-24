import requests
import json
from typing import Dict, Any

# Import database and crud for webhook lookup
from api.database import get_db
from api import crud, models

def dispatch_webhook(webhook_id: int, payload: Dict[str, Any]):
    """
    Dispatches a webhook by sending an HTTP POST request to its URL.
    """
    db = next(get_db())
    webhook = crud.get_webhook_by_id(db, webhook_id)
    db.close()

    if not webhook or not webhook.is_active:
        print(f"Webhook {webhook_id} not found or is inactive. Skipping dispatch.")
        return

    try:
        print(f"Dispatching webhook {webhook_id} to {webhook.url} with payload: {payload}")
        response = requests.post(webhook.url, json=payload, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes
        print(f"Webhook {webhook_id} dispatched successfully. Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error dispatching webhook {webhook_id} to {webhook.url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while dispatching webhook {webhook_id}: {e}")
