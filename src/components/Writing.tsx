import React from 'react';
import { Link } from 'react-router-dom';
import { posts } from '../data';
import { Reveal } from './Reveal';
import { ArrowRight } from 'lucide-react';

export const Writing: React.FC = () => {
  const recentPosts = posts.slice(0, 3);

  return (
    <section id="writing" className="py-24 bg-neutral-50 scroll-mt-24">
      <div className="container mx-auto px-6 max-w-4xl">
        <div className="flex items-center justify-between mb-16">
          <Reveal>
            <h2 className="text-3xl md:text-4xl font-heading font-bold text-neutral-900 tracking-tight">
              Latest Writing
            </h2>
          </Reveal>
          <Reveal delay={0.1}>
             <Link to="/writing" className="hidden md:flex items-center text-neutral-500 hover:text-neutral-900 transition-colors">
              View all posts <ArrowRight size={20} className="ml-2" />
            </Link>
          </Reveal>
        </div>

        <div className="space-y-8">
          {recentPosts.map((post, index) => (
            <Reveal key={post.id} width="100%" delay={index * 0.1}>
              <Link to={`/writing/${post.id}`} className="group block border-b border-neutral-200 pb-8 hover:border-neutral-400 transition-colors">
                <div className="flex flex-col md:flex-row md:items-baseline justify-between mb-3">
                  <h3 className="text-xl font-heading font-bold text-neutral-900 group-hover:text-blue-600 transition-colors">
                    {post.title}
                  </h3>
                  <span className="text-sm text-neutral-400 font-medium mt-1 md:mt-0">
                    {post.date}
                  </span>
                </div>
                <p className="text-neutral-500 leading-relaxed max-w-2xl">
                  {post.excerpt}
                </p>
              </Link>
            </Reveal>
          ))}
        </div>
        
        <div className="mt-8 md:hidden">
           <Link to="/writing" className="flex items-center text-neutral-500 hover:text-neutral-900 transition-colors">
              View all posts <ArrowRight size={20} className="ml-2" />
            </Link>
        </div>
      </div>
    </section>
  );
};