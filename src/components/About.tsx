import React from 'react';
import { Link } from 'react-router-dom';
import { Reveal } from './Reveal';
import { profile } from '../data';

export const About: React.FC = () => {
   return (
      <section id="about" className="py-24 bg-white scroll-mt-24">
         <div className="container mx-auto px-6 max-w-6xl">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-16 items-center">
               <Reveal>
                  <div className="relative">
                     <div className="aspect-[3/4] rounded-3xl overflow-hidden shadow-2xl">
                        <img
                           src={profile.profileImage}
                           alt={profile.name}
                           className="w-full h-full object-cover"
                        />
                     </div>
                  </div>
               </Reveal>

               <div>
                  <Reveal delay={0.2}>
                     <span className="text-blue-600 font-semibold tracking-wider uppercase text-sm mb-4 block">
                        About Me
                     </span>
                     <h2 className="text-4xl font-heading font-bold text-neutral-900 mb-8 leading-tight">
                        {profile.hero.heading} {profile.hero.rotatingTexts[0]}
                     </h2>
                  </Reveal>

                  <Reveal delay={0.3}>
                     <div className="space-y-6 text-lg text-neutral-500 leading-relaxed">
                        <p>
                           {profile.about.short}
                        </p>
                     </div>
                  </Reveal>

                  <Reveal delay={0.4}>
                     <div className="mt-10">
                        <Link
                           to="/about"
                           className="inline-flex items-center px-6 py-3 bg-neutral-900 text-white font-medium rounded-full hover:bg-neutral-800 transition-colors shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all"
                        >
                           More about me
                        </Link>
                     </div>
                  </Reveal>
               </div>
            </div>
         </div>
      </section>
   );
};
