from pydantic import BaseModel

class UserSubscription(BaseModel):
    is_subscribed: bool = False
    summaries_this_month: int = 0
    last_summary_date: str = "" # YYYY-MM-DD format
