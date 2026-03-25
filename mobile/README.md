# AAD Mobile

This is a minimal Expo React Native client for the existing Flask API.

What it does:

- lets you choose one image from the device gallery
- sends that image to `POST /api/detect_ai`
- shows the returned verdict and detector scores

This mobile app is intentionally separate from the deployed Flask web app. The current Render deployment still uses:

- `requirements-web.txt`
- `render.yaml`
- `Procfile`
- `gunicorn web.app:app`

That means adding this folder does not change the existing deployment path.

## Run locally

1. Install Node.js 20 LTS or newer.
2. Open a terminal in `mobile/`.
3. Run `npm install`.
4. Run `npx expo start`.

## Backend URL

Inside the app, set the backend URL to either:

- your deployed Render URL, or
- your machine's LAN IP such as `http://192.168.1.10:5000` when the Flask app is running locally

Do not use `http://localhost:5000` on a physical phone because `localhost` will point to the phone itself.
