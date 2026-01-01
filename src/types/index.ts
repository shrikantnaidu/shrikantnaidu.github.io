
export interface Project {
  id: string;
  title: string;
  category: string;
  description: string;
  fullContent?: string;
  imageUrl: string;
  link: string; // External link
  date: string;
  client?: string;
  tags?: string[];
}

export interface Post {
  id: string;
  title: string;
  date: string;
  excerpt: string;
  content?: string;
  category: string;
}

export interface Testimonial {
  id: string;
  quote: string;
  author: string;
  role: string;
}

export interface NavItem {
  label: string;
  href: string;
}

export interface Achievement {
  url: string;
  caption: string;
}

export interface Profile {
  name: string;
  role: string;
  email: string;
  location: string;
  social: {
    github: string;
    linkedin: string;
    twitter: string;
  };
  hero: {
    heading: string; // The main static text (e.g. "Architecting")
    rotatingTexts: string[];
    subheading: string;
  };
  about: {
    short: string; // Used on Home page (shortBio from markdown)
    long: string; // Markdown content for About page
  };
  tagline: string; // Blue text on About page
  profileImage: string; // Profile photo URL
  achievements: Achievement[]; // Slider images
}
