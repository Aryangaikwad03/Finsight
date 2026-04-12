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
        return {"category": "Other", "subcategory": "Other"}
        
    try:
        classifier = get_classifier()
        # Cap length for distilbert models to avoid length errors
        result = classifier(narration[:200]) 
        label = result[0]['label']
        
        # Attempt to map model's label directly to our app's broader buckets
        cat_map = {
            "food": ("Expense", "Food"),
            "dining": ("Expense", "Food"),
            "grocer": ("Expense", "Groceries"),
            "shopping": ("Expense", "Shopping"),
            "travel": ("Expense", "Travel"),
            "transportation": ("Expense", "Travel"),
            "util": ("Expense", "Utilities"),
            "bill": ("Expense", "Utilities"),
            "medic": ("Expense", "Healthcare"),
            "health": ("Expense", "Healthcare"),
            "entertainment": ("Expense", "Entertainment"),
            "recreation": ("Expense", "Entertainment"),
            "rent": ("Expense", "Rent"),
            "salary": ("Income", "Salary"),
            "invest": ("Investment", "Investment"),
        }
        
        lower_label = label.lower()
        matched_category = "Expense" # Default most things to Expense
        matched_sub = label # Keep the raw label for granularity
        
        for k, v in cat_map.items():
            if k in lower_label:
                matched_category = v[0]
                matched_sub = v[1]
                break
                
        return {"category": matched_category, "subcategory": matched_sub}
    except Exception as e:
        logger.error(f"Classification failed for '{narration}': {e}")
        return {"category": "Other", "subcategory": "Other"}
