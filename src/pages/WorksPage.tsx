import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { projects } from '../data';
import { Reveal } from '../components/Reveal';
import { ProjectCard } from '../components/ProjectCard';

export const WorksPage: React.FC = () => {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div className="min-h-screen pt-32 pb-24 bg-neutral-50">
      <div className="container mx-auto px-6">
        <div className="mb-16">
          <Link to="/" className="inline-flex items-center text-neutral-500 hover:text-neutral-900 mb-8 transition-colors font-medium">
            <ArrowLeft size={20} className="mr-2" /> Back to Home
          </Link>
          <Reveal>
            <h1 className="text-4xl md:text-5xl font-heading font-bold text-neutral-900 tracking-tight">
              All Projects
            </h1>
          </Reveal>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {projects.map((project, index) => (
            <Reveal key={project.id} delay={index * 0.05} width="100%" height="100%">
              <ProjectCard project={project} />
            </Reveal>
          ))}
        </div>
      </div>
    </div>
  );
};
