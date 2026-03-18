# 🚀 Streamlit Cloud Deployment Guide

## Quick Deploy Checklist

- [ ] Code pushed to GitHub
- [ ] Secrets configured
- [ ] Database file handled
- [ ] App deployed and tested
- [ ] Custom domain (optional)

---

## Step 1: Prepare Your Repository

### A. Create .streamlit/secrets.toml (LOCAL ONLY - Don't commit!)

```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "gsk_your_actual_key_here"
```

Add to `.gitignore`:
```
.streamlit/secrets.toml
```

### B. Update main.py to read from secrets

Replace:
```python
api_key = st.sidebar.text_input(label="🔑 Groq API Key", type="password")
```

With:
```python
# Try to get from secrets first, fallback to input
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    api_key = st.sidebar.text_input(label="🔑 Groq API Key", type="password")
```

---

## Step 2: Handle Database File

**Option A: Commit the database (RECOMMENDED for demo)**

```bash
# Remove *.db from .gitignore temporarily
git add retail_business.db
git commit -m "Add sample database"
git push
```

**Option B: Generate on startup**

Add to main.py (at the top):
```python
import subprocess
dbpath = Path(__file__).parent / "retail_business.db"
if not dbpath.exists():
    subprocess.run(["python", "create_retail_db.py"])
```

---

## Step 3: Push to GitHub

```bash
# Initialize git (if not done)
git init

# Add all files
git add .

# Commit
git commit -m "SQL Chat Bot - Production Ready"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/yourusername/sql-chat-bot.git

# Push
git push -u origin main
```

---

## Step 4: Deploy on Streamlit Cloud

### A. Go to Streamlit Cloud
1. Visit: https://share.streamlit.io
2. Click **"New app"**
3. Sign in with GitHub

### B. Configure App

**Repository:** Select your GitHub repo

**Branch:** `main`

**Main file path:** `main.py`

### C. Add Secrets

Click **"Advanced settings"** → **"Secrets"**

Add:
```toml
GROQ_API_KEY = "gsk_your_actual_groq_key_here"
```

### D. Deploy!

Click **"Deploy!"**

Wait 2-3 minutes for deployment.

---

## Step 5: Test Your Deployment

### Checklist:
- [ ] App loads without errors
- [ ] Sample questions work
- [ ] Database queries execute
- [ ] Results display correctly
- [ ] No API key input needed (using secrets)
- [ ] Chat history works
- [ ] Error handling works

### Test Queries:
1. "What are the top 5 best-selling products?"
2. "Show total revenue"
3. "Which products are low in stock?"
4. Try an invalid query to test error handling

---

## Step 6: Get Your Deployment URL

Your app will be at:
```
https://[your-app-name].streamlit.app
```

**Share this URL on:**
- GitHub README (add badge)
- LinkedIn post
- Resume
- Portfolio website

---

## Optional: Custom Domain

### Add Custom Subdomain

In Streamlit Cloud settings:
1. Go to **Settings** → **General**
2. Update **App URL**
3. Choose custom subdomain: `sql-chat-bot.streamlit.app`

---

## Troubleshooting

### "Module not found" error
- Check `requirements.txt` has all dependencies
- Verify version numbers are correct

### "Database not found" error
- Ensure `retail_business.db` is in repo
- OR `create_retail_db.py` runs on startup

### "API key not found" error
- Check secrets are configured correctly
- Verify secret name matches: `GROQ_API_KEY`

### App is slow
- Add caching to database connection
- Already done with `@st.cache_resource`

---

## Updating Your Deployed App

```bash
# Make changes locally
# Test locally: streamlit run main.py

# Commit and push
git add .
git commit -m "Update: [describe changes]"
git push

# Streamlit Cloud auto-deploys on push!
```

---

## Monitoring Your App

### View Logs:
1. Go to https://share.streamlit.io
2. Click on your app
3. Click **"Manage app"** → **"Logs"**

### Analytics:
- Streamlit Cloud shows basic usage stats
- View count, active users, etc.

---

## Production Checklist

Before sharing widely:

- [ ] Test all sample questions
- [ ] Test error scenarios
- [ ] Verify mobile responsiveness
- [ ] Check loading speed
- [ ] Update README with live demo link
- [ ] Add screenshots to README
- [ ] Create demo video (optional)
- [ ] Share on LinkedIn/Twitter

---

## Next Steps

1. **Deploy NOW** (don't wait for perfection)
2. Test thoroughly
3. Share with 3-5 friends for feedback
4. Iterate based on feedback
5. Add to resume with deployment link

**Your live demo link is your most valuable asset!** 🚀
