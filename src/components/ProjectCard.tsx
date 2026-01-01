import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import { Project } from '../types';

interface ProjectCardProps {
  project: Project;
}

export const ProjectCard: React.FC<ProjectCardProps> = ({ project }) => {
  return (
    <Link
      to={`/works/${project.id}`}
      className="group block bg-white rounded-2xl overflow-hidden shadow-sm hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 h-full flex flex-col"
    >
      <div className="relative aspect-[4/3] overflow-hidden">
        <img
          src={project.imageUrl}
          alt={project.title}
          className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
        />
        <div className="absolute inset-0 bg-neutral-900/0 group-hover:bg-neutral-900/5 transition-colors duration-300" />
      </div>
      <div className="p-8 flex flex-col flex-grow">
        <div className="mb-4">
          <span className="text-xs font-bold text-blue-600 uppercase tracking-wider">
            {project.category}
          </span>
        </div>
        <h3 className="text-xl font-heading font-bold text-neutral-900 mb-3 group-hover:text-blue-600 transition-colors">
          {project.title}
        </h3>
        <p className="text-neutral-500 leading-relaxed mb-4 line-clamp-2 flex-grow">
          {project.description}
        </p>

        {/* Tags */}
        {project.tags && project.tags.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-6 mt-auto">
            {project.tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                className="px-2 py-1 bg-neutral-100 text-neutral-600 text-[10px] font-bold uppercase tracking-wider rounded"
              >
                {tag}
              </span>
            ))}
            {project.tags.length > 3 && (
               <span className="px-2 py-1 bg-neutral-100 text-neutral-400 text-[10px] font-bold uppercase tracking-wider rounded">+{project.tags.length - 3}</span>
            )}
          </div>
        )}

        <div className={`flex items-center text-sm font-bold text-neutral-900 ${(!project.tags || project.tags.length === 0) ? 'mt-auto' : ''}`}>
          View <ArrowRight size={16} className="ml-2 group-hover:translate-x-1 transition-transform" />
        </div>
      </div>
    </Link>
  );
};
