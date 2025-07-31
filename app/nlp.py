def handle_user_question(question):
    q = question.lower()
    if "jean" in q and "wash" in q:
        return {"tip": "Wash jeans in cold water and air dry for longevity."}
    if "hole" in q and "repair" in q:
        return {"tip": "Use an iron-on patch or hand-stitch small holes to extend garment life."}
    return {"tip": "Try to repair, upcycle or donate clothing for sustainable fashion!"}
