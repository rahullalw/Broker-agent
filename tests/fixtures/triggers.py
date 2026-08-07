"""The ~30-phrase trigger suite (Phase 4 §2).

**Negatives first, deliberately.** Every false positive permanently silences the
agent for that buyer — the lead is handed to a broker who was never going to be
called, and the qualification arc the product exists to run never happens. A
false negative costs one missed handoff; a false positive costs the buyer.

English, Hinglish, and the Hinglish an Ahmedabad buyer actually types.
"""

MUST_NOT_ESCALATE = [
    # --- the six named in the spec -------------------------------------
    "loan lunga, budget 65L",
    "kam se kam 3BHK chahiye",
    "ye rate market rate hai kya",
    "builder ne discount diya tha pichle project me",
    "accurate carpet area batao",
    "corporate lease pe hai kya",

    # --- the three seeded openings must all survive --------------------
    "Hi, looking for a 3BHK in Bopal",
    "3bhk chahiye bhai, budget 65 lakh",
    "3BHK in Satellite under 80 lakh",

    # --- ordinary qualification that mentions a trigger word -----------
    "home loan le raha hoon, 65 lakh tak ka budget hai",
    "EMI ke bare me baad me dekhenge, pehle property dikhao",
    "legal team ne check kiya tha pichli baar",

    # --- asking what something costs is not asking for a discount ------
    "price kya hai is flat ka",
    "kitne ka hai ye flat",
    "iska rate accurate batao",

    # --- plain questions -----------------------------------------------
    "possession kab tak milega?",
    "carpet area kitna hai",
    "Satellite me koi ready to move hai?",
]

MUST_ESCALATE = [
    # --- the four named in the spec ------------------------------------
    "price kam karo bhai",
    "thoda discount milega?",
    "EMI kitna banega",
    "aadmi se baat karao",

    # --- negotiation ----------------------------------------------------
    "kuch kam hoga kya?",
    "best rate batao",
    "mol bhav ho sakta hai?",
    "final price batao",
    "can we negotiate the price?",
    "rate kam kar do",

    # --- loan advice ----------------------------------------------------
    "loan dila do bhai",
    "interest rate kya hai?",
    "kaunsa bank loan dega?",

    # --- legal / dispute -------------------------------------------------
    "ye to dhokha hai, legal notice bhejunga",
    "RERA me complaint karunga",

    # --- asking for a human ----------------------------------------------
    "mujhe manager se baat karni hai",
    "I want to talk to a real person",
]
