import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { posts } from '../data';
import { Reveal } from '../components/Reveal';

export const WritingPage: React.FC = () => {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div className="min-h-screen pt-32 pb-24 bg-white">
      <div className="container mx-auto px-6 max-w-4xl">
        <div className="mb-16">
          <Link to="/" className="inline-flex items-center text-neutral-500 hover:text-neutral-900 mb-8 transition-colors font-medium">
            <ArrowLeft size={20} className="mr-2" /> Back to Home
          </Link>
          <Reveal>
            <h1 className="text-4xl md:text-5xl font-heading font-bold text-neutral-900 tracking-tight mb-4">
              Writing
            </h1>
            <p className="text-xl text-neutral-500">Thoughts on engineering, leadership, and data.</p>
          </Reveal>
        </div>

        <div className="space-y-8">
          {posts.map((post, index) => (
            <Reveal key={post.id} width="100%" delay={index * 0.1}>
              <Link
                to={`/writing/${post.id}`}
                className="group block p-8 rounded-2xl bg-neutral-50 hover:bg-white border border-transparent hover:border-neutral-200 hover:shadow-lg transition-all duration-300"
              >
                <div className="flex flex-col md:flex-row md:items-baseline justify-between mb-4">
                  <h3 className="text-2xl font-heading font-bold text-neutral-900 group-hover:text-blue-600 transition-colors">
                    {post.title}
                  </h3>
                  <span className="text-sm text-neutral-400 font-medium mt-2 md:mt-0">
                    {post.date}
                  </span>
                </div>
                <div className="mb-4">
                    <span className="inline-block px-3 py-1 bg-white rounded-full text-xs font-semibold text-neutral-600 border border-neutral-100">
                        {post.category}
                    </span>
                </div>
                <p className="text-neutral-500 leading-relaxed">
                  {post.excerpt}
                </p>
                <div className="mt-6 flex items-center text-blue-600 font-medium opacity-0 group-hover:opacity-100 transition-opacity">
                    Read Article <ArrowRight size={16} className="ml-2" />
                </div>
              </Link>
            </Reveal>
          ))}
        </div>
      </div>
    </div>
  );
};