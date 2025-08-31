from .database import get_firestore_db # Import the function to get the db client
from .models import UserSubscription
from datetime import datetime

async def get_user_subscription(uid: str) -> UserSubscription:
    user_ref = get_firestore_db().collection('users').document(uid)
    doc = await user_ref.get()
    if doc.exists:
        data = doc.to_dict()
        # Reset monthly count if new month
        today = datetime.now().strftime("%Y-%m-%d")
        if data.get("last_summary_date") and datetime.strptime(data["last_summary_date"], "%Y-%m-%d").month != datetime.now().month:
            data["summaries_this_month"] = 0
            data["last_summary_date"] = today
            await user_ref.set(data) # Update Firestore
        return UserSubscription(**data)
    else:
        # Create a new entry for the user if they don't exist
        new_user_data = UserSubscription().dict()
        new_user_data["last_summary_date"] = datetime.now().strftime("%Y-%m-%d")
        await user_ref.set(new_user_data)
        return UserSubscription(**new_user_data)

async def update_user_subscription(uid: str, data: dict):
    user_ref = get_firestore_db().collection('users').document(uid)
    await user_ref.update(data)

async def increment_summary_count(uid: str):
    user_ref = get_firestore_db().collection('users').document(uid)
    doc = await user_ref.get()
    if doc.exists:
        data = doc.to_dict()
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Reset monthly count if new month
        if data.get("last_summary_date") and datetime.strptime(data["last_summary_date"], "%Y-%m-%d").month != datetime.now().month:
            data["summaries_this_month"] = 1
            data["last_summary_date"] = today
        else:
            # Increment count if same month
            data["summaries_this_month"] = data.get("summaries_this_month", 0) + 1
        
        await user_ref.update(data) # Persist changes to Firestore
    else:
        # This case should ideally not happen if get_user_subscription is called first,
        # but as a fallback, create a new entry.
        new_user_data = UserSubscription().dict()
        new_user_data["last_summary_date"] = datetime.now().strftime("%Y-%m-%d")
        new_user_data["summaries_this_month"] = 1
        await user_ref.set(new_user_data)
        
