import React from 'react';
import { ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { projects } from '../data';
import { Reveal } from './Reveal';
import { ProjectCard } from './ProjectCard';

export const Works: React.FC = () => {
  const featuredProjects = projects.slice(0, 3);

  return (
    <section id="works" className="py-16 md:py-24 bg-neutral-50 scroll-mt-24">
      <div className="container mx-auto px-6">
        <div className="flex items-end justify-between mb-16">
          <Reveal>
            <h2 className="text-3xl md:text-4xl font-heading font-bold text-neutral-900 tracking-tight">
              Selected Works
            </h2>
          </Reveal>
          <Reveal delay={0.1}>
            <Link to="/works" className="hidden md:flex items-center text-neutral-500 hover:text-neutral-900 transition-colors">
              View all projects <ArrowRight size={20} className="ml-2" />
            </Link>
          </Reveal>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {featuredProjects.map((project, index) => (
            <Reveal key={project.id} delay={index * 0.1} width="100%" height="100%">
              <ProjectCard project={project} />
            </Reveal>
          ))}
        </div>

        <div className="mt-12 md:hidden">
          <Reveal delay={0.2}>
            <Link to="/works" className="flex items-center text-neutral-500 hover:text-neutral-900 transition-colors">
              View all projects <ArrowRight size={20} className="ml-2" />
            </Link>
          </Reveal>
        </div>
      </div>
    </section>
  );
};
