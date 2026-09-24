from main import classify_question, is_analysis_request

q = "i want a full analysis for the last 3 months in 2025"
print("keyword match:", is_analysis_request(q))
print("classify_question:", classify_question(q))