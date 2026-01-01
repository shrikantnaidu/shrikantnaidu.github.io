
import React, { useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, ExternalLink, Calendar, Briefcase } from 'lucide-react';
import Markdown from 'react-markdown';
import { projects } from '../data';
import { Reveal } from '../components/Reveal';

export const ProjectPage: React.FC = () => {
   const { id } = useParams<{ id: string }>();
   const project = projects.find(p => p.id === id);

   useEffect(() => {
      window.scrollTo(0, 0);
   }, [id]);

   if (!project) {
      return (
         <div className="min-h-screen flex items-center justify-center">
            <div className="text-center">
               <h2 className="text-2xl font-bold mb-4">Project not found</h2>
               <Link to="/works" className="text-blue-600 hover:underline">Back to projects</Link>
            </div>
         </div>
      );
   }

   return (
      <article className="min-h-screen pt-32 pb-24 bg-white">
         <div className="container mx-auto px-6 max-w-5xl">
            <Link to="/works" className="inline-flex items-center text-neutral-500 hover:text-neutral-900 mb-8 transition-colors font-medium">
               <ArrowLeft size={20} className="mr-2" /> Back to Projects
            </Link>

            <Reveal>
               <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-6">
                  <h1 className="text-4xl md:text-6xl font-heading font-bold text-neutral-900 leading-tight">
                     {project.title}
                  </h1>
                  <span className="inline-block px-4 py-2 bg-blue-50 text-blue-700 font-semibold rounded-full text-sm">
                     {project.category}
                  </span>
               </div>
               <p className="text-xl md:text-2xl text-neutral-500 leading-relaxed mb-12 max-w-3xl">
                  {project.description}
               </p>
            </Reveal>

            {/* Image and Details Side by Side */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
               {/* Main Image */}
               <div className="lg:col-span-2">
                  <div className="rounded-3xl overflow-hidden shadow-2xl ring-1 ring-neutral-100 aspect-video bg-gradient-to-br from-blue-100 to-purple-100">
                     <img
                        src={project.imageUrl}
                        alt={project.title}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                           (e.target as HTMLImageElement).style.display = 'none';
                        }}
                     />
                  </div>
               </div>

               {/* Details Card */}
               <div className="lg:col-span-1">
                  <div className="bg-neutral-50 rounded-2xl p-6 border border-neutral-100 h-full flex flex-col">
                     <h3 className="font-heading font-bold text-lg mb-4 text-neutral-900">Project Details</h3>
                     <div className="space-y-4 flex-grow">
                        <div>
                           <span className="block text-xs font-bold text-neutral-400 uppercase tracking-wider mb-1">Client</span>
                           <span className="text-neutral-900 font-medium">{project.client || 'Confidential'}</span>
                        </div>
                        <div>
                           <span className="block text-xs font-bold text-neutral-400 uppercase tracking-wider mb-1">Date</span>
                           <div className="flex items-center text-neutral-900 font-medium">
                              <Calendar size={14} className="mr-1 text-neutral-400" />
                              {new Date(project.date).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })}
                           </div>
                        </div>
                        <div>
                           <span className="block text-xs font-bold text-neutral-400 uppercase tracking-wider mb-1">Role</span>
                           <div className="flex items-center text-neutral-900 font-medium">
                              <Briefcase size={14} className="mr-1 text-neutral-400" />
                              Lead ML Engineer
                           </div>
                        </div>

                        {/* Technologies Tags */}
                        {project.tags && project.tags.length > 0 && (
                           <div className="pt-3 border-t border-neutral-200">
                              <span className="block text-xs font-bold text-neutral-400 uppercase tracking-wider mb-2">Technologies</span>
                              <div className="flex flex-wrap gap-1.5">
                                 {project.tags.map(tag => (
                                    <span key={tag} className="px-2 py-1 bg-white border border-neutral-200 text-neutral-600 text-xs rounded font-medium">
                                       {tag}
                                    </span>
                                 ))}
                              </div>
                           </div>
                        )}
                     </div>

                     <a
                        href={project.link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="mt-4 inline-flex items-center justify-center w-full px-4 py-2.5 bg-neutral-900 text-white font-medium rounded-xl hover:bg-neutral-800 transition-colors text-sm"
                     >
                        View Project <ExternalLink size={14} className="ml-2" />
                     </a>
                  </div>
               </div>
            </div>

            {/* Main Content - No animation delay */}
            <div className="prose prose-lg prose-neutral prose-headings:font-heading prose-a:text-blue-600 max-w-4xl mx-auto">
               <Markdown>{project.fullContent}</Markdown>
            </div>
         </div>
      </article>
   );
};
