<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1Fmm17tJOrO8c6jHfuJS3xuzD5b8F4gdj

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

## Google Analytics
To enable Google Analytics tracking:
1. Get your **Measurement ID** (e.g., `G-XXXXXXXXXX`) from the Google Analytics dashboard.
2. Add it to your `.env.local` file:
   ```
   VITE_GA_MEASUREMENT_ID=your_id_here
   ```
