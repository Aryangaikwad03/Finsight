import logging
from transformers import pipeline

logger = logging.getLogger("finsight.categorizer")

_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        logger.info("Initializing HuggingFace classification pipeline (mitulshah/global-financial-transaction-classifier)...")
        _classifier = pipeline("text-classification", model="mitulshah/global-financial-transaction-classifier")
        logger.info("Model loaded successfully.")
    return _classifier

def categorize_transaction_ml(narration: str) -> dict:
    if not narration or len(narration.strip()) < 3:
        return "Other"
        
    try:
        classifier = get_classifier()
        # Cap length for distilbert models to avoid length errors
        result = classifier(narration[:200]) 
        label = result[0]['label']
        
        # Attempt to map model's label directly to our app's broader buckets
        cat_map = {
            "food": "Food & Dining",
            "dining": "Food & Dining",
            "grocer": "Groceries",
            "shopping": "Shopping & Retail",
            "travel": "Travel",
            "transportation": "Travel",
            "util": "Bills & Utilities",
            "bill": "Bills & Utilities",
            "medic": "Healthcare & Medical",
            "health": "Healthcare & Medical",
            "entertainment": "Entertainment & Leisure",
            "recreation": "Entertainment & Leisure",
            "rent": "Housing & Rent",
            "salary": "Salary & Income",
            "invest": "Investments & Savings",
        }
        
        lower_label = label.lower()
        matched_category = "Uncategorized / Unknown" # Default most things to Expense
         # Keep the raw label for granularity
        
        for k, v in cat_map.items():
            if k in lower_label:
                matched_category = v
                break
                
        return matched_category
    except Exception as e:
        logger.error(f"Classification failed for '{narration}': {e}")
        return "Other"
