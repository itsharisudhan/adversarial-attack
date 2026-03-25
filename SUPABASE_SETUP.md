# Supabase Setup

This app now supports optional analysis history persistence for uploaded and analyzed images.

## Behavior

- If `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are not set, the app behaves exactly as before.
- If both are set, the app saves lightweight history records asynchronously after each analysis.
- Persistence failures do not block image analysis.

## 1. Create the Table

Run the SQL in [web/supabase_schema.sql](c:\Users\Hari2025\Desktop\adversarial-attack-master\adversarial-attack-master\web\supabase_schema.sql) inside the Supabase SQL editor.

## 2. Add Environment Variables

Set these on your deployment platform and locally when needed:

```text
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_SERVER_KEY=YOUR_SERVER_SIDE_SUPABASE_KEY
SUPABASE_HISTORY_TABLE=analysis_history
```

`SUPABASE_SERVER_KEY` can be either:
- the newer `sb_secret_...` secret key, or
- the older `service_role` key

Use it only on the server side. Do not expose it in frontend code.

## 3. Local Run

PowerShell example:

```powershell
$env:SUPABASE_URL="https://YOUR_PROJECT.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY="YOUR_SERVICE_ROLE_KEY"
python web\app.py
```

## What Gets Stored

- Filename
- Verdict and score
- Small input preview
- FFT spectrum preview
- ELA heatmap preview
- Detector score summary
- Image size metadata

This keeps the integration lightweight and avoids storing full original files.
