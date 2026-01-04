# Shrikant Naidu | Personal Portfolio

A modern, high-fidelity personal portfolio built with **React**, **Vite**, and **Tailwind CSS**. Designed with a focus on typography, smooth transitions, and a developer-friendly "Content-as-Data" workflow.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   npm install
   ```
2. **Launch development server:**
   ```bash
   npm run dev
   ```
3. **Build for production:**
   ```bash
   npm run build
   ```

## 🛠 Project Structure & Configuration

This project uses a "Jekyll-style" workflow where the UI is driven by Markdown files. You can update your bio, social links, and project details without touching the source code.

### 1. Profile & About
Modify **`src/content/about.md`** to update:
- **Hero Text**: Headline and rotating subheadings.
- **Social Links**: GitHub, LinkedIn, Twitter, Steam, Hugging Face, etc.
- **Achievements**: Images and captions for the "Awards" slider on the About page.
- **Contact Form**: Set your **Formspree ID** directly in the YAML front-matter.

### 2. Projects & Writing
Update **`src/content/projects/`** or **`src/content/posts/`**:
- Each `.md` file represents a project or post.
- Uses front-matter for metadata (title, date, cover images, tags).

## 📬 Contact Form Setup

The contact form is powered by **Formspree**. To make it functional:
1. Create a form at [Formspree](https://formspree.io/).
2. Copy your unique Form ID (or the full endpoint URL).
3. Paste it in `src/content/about.md` under the `contact` key:
   ```yaml
   contact:
     formspreeId: "your_id_here"
   ```

## 📊 Analytics

To enable Google Analytics tracking, add your Measurement ID to your `.env.local` file:
```env
VITE_GA_MEASUREMENT_ID=G-XXXXXXXXXX
```

## 🎨 Tech Stack
- **React 18** + **Vite**
- **Tailwind CSS** for styling
- **Lucide React** for iconography
- **React Markdown** for content rendering
- **Front-matter** for Jekyll-style data management
