import fm from 'front-matter';
import { Project, Post, Testimonial, Profile, Achievement } from '../types';

// --- About/Profile Loader from Markdown ---
interface AboutAttributes {
  name: string;
  role: string;
  email: string;
  location: string;
  social: {
    github: string;
    linkedin: string;
    twitter: string;
    lightning?: string;
    wandb?: string;
    huggingface?: string;
    steam?: string;
  };
  hero: {
    heading: string;
    subheading: string;
    rotatingTexts: string[];
  };
  shortBio: string;
  tagline: string;
  profileImage: string;
  achievements: Achievement[];
  contact: {
    formspreeId: string;
  };
}

const aboutFile = import.meta.glob('/src/content/about.md', {
  eager: true,
  query: '?raw',
  import: 'default'
});

const aboutPath = Object.keys(aboutFile)[0];
const aboutContent = aboutPath ? (aboutFile[aboutPath] as string) : '';

let profile: Profile;

if (aboutContent) {
  const parse = (fm as any).default || fm;
  const parsed = parse(aboutContent.trim());
  const attrs = parsed.attributes as AboutAttributes;

  profile = {
    name: attrs.name,
    role: attrs.role,
    email: attrs.email,
    location: attrs.location,
    social: attrs.social,
    hero: attrs.hero,
    about: {
      short: attrs.shortBio,
      long: parsed.body // Full markdown content
    },
    tagline: attrs.tagline,
    profileImage: attrs.profileImage,
    achievements: attrs.achievements || [],
    contact: attrs.contact || { formspreeId: "" }
  };

  console.log('Profile loaded from about.md:', profile.name);
} else {
  // Fallback if about.md is not found
  console.warn('about.md not found, using fallback profile');
  profile = {
    name: "Your Name",
    role: "Your Role",
    email: "email@example.com",
    location: "Location",
    social: {
      github: "",
      linkedin: "",
      twitter: "",
      lightning: "",
      wandb: "",
      huggingface: "",
      steam: "",
    },
    hero: {
      heading: "Building",
      rotatingTexts: ["Amazing Things."],
      subheading: "Welcome to my portfolio"
    },
    about: {
      short: "Add your short bio here.",
      long: "Add your detailed bio here."
    },
    tagline: "Your Tagline Here.",
    profileImage: "",
    achievements: [],
    contact: {
      formspreeId: "",
    },
  };
}

export { profile };


// --- Dynamic Project Loader ---

interface ProjectAttributes {
  title: string;
  category: string;
  description: string;
  imageUrl: string;
  link: string;
  date: string | Date;
  client?: string;
  tags?: string[];
}

const projectFiles = import.meta.glob('/src/projects/*.md', {
  eager: true,
  query: '?raw',
  import: 'default'
});

console.log('Project files found:', Object.keys(projectFiles));

const dynamicProjects: Project[] = Object.keys(projectFiles).map((path): Project | null => {
  try {
    const rawContent = projectFiles[path] as string;
    const content = rawContent.trim();

    console.log(`--- Processing: ${path} ---`);
    console.log('Raw content first 100 chars:', JSON.stringify(rawContent.substring(0, 100)));
    console.log('Trimmed content first 100 chars:', JSON.stringify(content.substring(0, 100)));

    // Handle potential CJS/ESM interop issues with front-matter
    const parse = (fm as any).default || fm;
    console.log('Parse function type:', typeof parse);

    const parsed = parse(content);
    console.log('Parsed attributes:', parsed.attributes);
    console.log('Parsed body length:', parsed.body?.length);

    const { title, category, description, imageUrl, link, date, client, tags } = parsed.attributes as ProjectAttributes;
    const id = path.split('/').pop()?.replace('.md', '') || 'unknown';

    const dateString = date instanceof Date ? date.toISOString().split('T')[0] : String(date);

    const project = {
      id,
      title,
      category,
      description,
      imageUrl,
      link,
      date: dateString,
      client,
      tags,
      fullContent: parsed.body
    };

    console.log('Created project:', { id, title, hasContent: !!parsed.body });

    return project;
  } catch (e) {
    console.error(`Error loading project ${path}:`, e);
    return null;
  }
}).filter((p): p is Project => p !== null);

// Sort projects by date (descending - newest first)
export const projects: Project[] = dynamicProjects.sort((a, b) => {
  const dateA = new Date(a.date).getTime() || 0;
  const dateB = new Date(b.date).getTime() || 0;
  return dateB - dateA;
});

console.log('Final Projects List:', projects);


// --- Dynamic Post Loader ---

interface PostAttributes {
  title: string;
  date: string | Date;
  category: string;
  excerpt: string;
}

const markdownFiles = import.meta.glob('/src/posts/*.md', {
  eager: true,
  query: '?raw',
  import: 'default'
});

export const posts: Post[] = Object.keys(markdownFiles).map((path): Post | null => {
  try {
    const content = markdownFiles[path] as string;
    const parse = (fm as any).default || fm;
    const parsed = parse(content);

    const { title, date, category, excerpt } = parsed.attributes as PostAttributes;
    const id = path.split('/').pop()?.replace('.md', '') || 'unknown';

    const dateString = date instanceof Date ? date.toLocaleDateString('en-US', {
      year: 'numeric', month: 'short', day: 'numeric'
    }) : String(date);

    return {
      id,
      title,
      date: dateString,
      category,
      excerpt,
      content: parsed.body
    };
  } catch (e) {
    console.error(`Error loading post ${path}:`, e);
    return null;
  }
}).filter((p): p is Post => p !== null)
  .sort((a, b) => {
    // Sort by date descending (newest first)
    const dateA = new Date(a.date).getTime() || 0;
    const dateB = new Date(b.date).getTime() || 0;
    return dateB - dateA;
  });

export const testimonials: Testimonial[] = [
  {
    id: '1',
    quote: "Shrikant bridges the gap between complex research and production code. The RAG system he built transformed how we access internal knowledge.",
    author: "Priya Mehta",
    role: "VP of Engineering, FinTech Corp",
  },
  {
    id: '2',
    quote: "The visual inspection model improved our yield by 15% and has been running 24/7 without downtime. A reliable and scalable solution.",
    author: "James Wilson",
    role: "Director of Ops, Manufacturing Inc.",
  },
];
