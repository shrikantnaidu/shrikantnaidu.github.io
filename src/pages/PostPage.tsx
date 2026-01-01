import React, { useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import Markdown from 'react-markdown';
import { posts } from '../data';
import { Reveal } from '../components/Reveal';

export const PostPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const post = posts.find(p => p.id === id);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [id]);

  if (!post) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-4">Post not found</h2>
          <Link to="/writing" className="text-blue-600 hover:underline">Back to writing</Link>
        </div>
      </div>
    );
  }

  return (
    <article className="min-h-screen pt-32 pb-24 bg-white">
      <div className="container mx-auto px-6 max-w-3xl">
        <Link to="/writing" className="inline-flex items-center text-neutral-500 hover:text-neutral-900 mb-12 transition-colors font-medium">
          <ArrowLeft size={20} className="mr-2" /> Back to Writing
        </Link>

        <Reveal width="100%">
          <div className="mb-10">
            <div className="flex items-center space-x-4 mb-6">
              <span className="text-sm font-semibold text-blue-600 bg-blue-50 px-3 py-1 rounded-full">
                {post.category}
              </span>
              <span className="text-sm text-neutral-400 font-medium">
                {post.date}
              </span>
            </div>
            <h1 className="text-4xl md:text-5xl font-heading font-extrabold text-neutral-900 leading-tight">
              {post.title}
            </h1>
          </div>
        </Reveal>

        <Reveal delay={0.2}>
          <div className="prose prose-lg prose-neutral prose-headings:font-heading prose-a:text-blue-600 hover:prose-a:text-blue-700 max-w-none">
            <p className="lead text-xl text-neutral-500 mb-8 font-normal leading-relaxed">
              {post.excerpt}
            </p>
            <Markdown>{post.content}</Markdown>
          </div>
        </Reveal>
      </div>
    </article>
  );
};