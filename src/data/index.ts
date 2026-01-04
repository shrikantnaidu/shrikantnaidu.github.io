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
    aboutHeading: string;
    subheading: string;
    rotatingTexts: string[];
  };
  shortBio: string;
  tagline: string;
  profileImage: string;
  achievements: Achievement[];
  contact: {
    formspreeId: string;
    googleAnalyticsId?: string;
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
    contact: {
      formspreeId: attrs.contact?.formspreeId || "",
      googleAnalyticsId: attrs.contact?.googleAnalyticsId
    }
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
      aboutHeading: "Turning complexity into working systems.",
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
      googleAnalyticsId: ""
    },
  };
}

export { profile };


// --- Dynamic Project Loader ---

interface ProjectAttributes {
  title: string;
  category: string;
  date: string;
  client?: string;
  description: string;
  imageUrl: string;
  link: string;
  tags?: string[];
}

const projectFiles = import.meta.glob('/src/projects/*.md', {
  eager: true,
  query: '?raw',
  import: 'default'
});

const projects: Project[] = Object.entries(projectFiles).map(([path, content]) => {
  const parse = (fm as any).default || fm;
  const { attributes, body } = parse((content as string).trim());
  const attrs = attributes as ProjectAttributes;

  // Extract ID from filename (e.g., 'data-modeling-with-postgres' from '/src/projects/data-modeling-with-postgres.md')
  const id = path.split('/').pop()?.replace('.md', '') || '';

  return {
    id,
    ...attrs,
    fullContent: body
  };
}).sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

export { projects };


// --- Dynamic Post Loader ---

interface PostAttributes {
  title: string;
  date: string;
  category: string;
}

const postFiles = import.meta.glob('/src/posts/*.md', {
  eager: true,
  query: '?raw',
  import: 'default'
});

const posts: Post[] = Object.entries(postFiles).map(([path, content]) => {
  const parse = (fm as any).default || fm;
  const { attributes, body } = parse((content as string).trim());
  const attrs = attributes as PostAttributes;

  const id = path.split('/').pop()?.replace('.md', '') || '';

  // Calculate excerpt from body
  const excerpt = body
    .replace(/[#*`]/g, '') // Remove markdown symbols
    .split('\n')
    .find((line: string) => line.trim().length > 0)
    ?.slice(0, 150) + '...';

  return {
    id,
    ...attrs,
    excerpt: excerpt || '',
    content: body
  };
}).sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

export { posts };

// --- Static Data ---
export const testimonials: Testimonial[] = [
  {
    id: '1',
    quote: "Working with this team was a game-changer for our data strategy.",
    author: "Sarah Johnson",
    role: "CTO at TechFlow"
  },
  {
    id: '2',
    quote: "The insight they provided helped us scale our infrastructure 10x.",
    author: "Michael Chen",
    role: "VP of Engineering at DataSystems"
  }
];

export const navItems = [
  { label: 'Home', href: '#home' },
  { label: 'Works', href: '#works' },
  { label: 'Writing', href: '#writing' },
  { label: 'About', href: '#about' },
  { label: 'Contact', href: '#contact' },
];
