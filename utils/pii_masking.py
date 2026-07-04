import re

# ---------------- EMAIL ----------------
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# ---------------- PHONE ----------------
PHONE_RE = re.compile(r"""
(?<!\w)
(\+?\d{1,3}[\s\-\.]?)?
(\(?\d{2,4}\)?[\s\-\.]?)?
\d{3,4}[\s\-\.]?\d{3,4}
(?!\w)
""", re.VERBOSE)

# ---------------- URL (general) ----------------
URL_RE = re.compile(r"https?://[^\s<>\"]+|www\.[^\s<>\"]+")

# ---------------- LINKEDIN (stronger) ----------------
LINKEDIN_RE = re.compile(
    r"(https?://)?(www\.)?linkedin\.com/[^\s]+",
    re.IGNORECASE
)

# ---------------- GITHUB (stronger) ----------------
GITHUB_RE = re.compile(
    r"(https?://)?(www\.)?github\.com/[^\s]+",
    re.IGNORECASE
)


# ---------------- MASK FUNCTION ----------------
def mask_pii(text: str) -> str:
    if not text:
        return text

    # EMAIL
    text = EMAIL_RE.sub("<EMAIL>", text)

    # PHONE
    text = PHONE_RE.sub("<PHONE>", text)

    # LINKEDIN
    text = LINKEDIN_RE.sub("<LINKEDIN>", text)

    # GITHUB
    text = GITHUB_RE.sub("<GITHUB>", text)

    # OTHER URLs (last step!)
    text = URL_RE.sub("<URL>", text)

    return text