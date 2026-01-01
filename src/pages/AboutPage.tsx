import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, MapPin, Mail, Linkedin, Github } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Reveal } from '../components/Reveal';
import { profile } from '../data';

export const AboutPage: React.FC = () => {
  const [currentSlide, setCurrentSlide] = useState(0);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  useEffect(() => {
    if (profile.achievements.length === 0) return;
    const timer = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % profile.achievements.length);
    }, 4000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="min-h-screen pt-32 bg-white">
      {/* Top Section with Bio and Sidebar */}
      <div className="container mx-auto px-6 max-w-6xl">
        <Link to="/" className="inline-flex items-center text-neutral-500 hover:text-neutral-900 mb-12 transition-colors font-medium">
          <ArrowLeft size={20} className="mr-2" /> Back to Home
        </Link>

        <div className="grid grid-cols-1 md:grid-cols-12 gap-16 mb-12">
          {/* Main Content */}
          <div className="md:col-span-8">
            <Reveal>
              <h1 className="text-4xl md:text-5xl font-heading font-bold text-neutral-900 mb-8 leading-tight">
                Hello, I'm {profile.name.split(' ')[0]}. <br />
                <span className="text-blue-600">{profile.tagline}</span>
              </h1>
            </Reveal>

            <Reveal delay={0.1}>
              <div className="prose prose-lg prose-neutral max-w-none text-neutral-600 leading-relaxed">
                <ReactMarkdown>{profile.about.long}</ReactMarkdown>
              </div>
            </Reveal>
          </div>

          {/* Sidebar */}
          <div className="md:col-span-4">
            <Reveal delay={0.2}>
              <div className="sticky top-32">
                {profile.profileImage && (
                  <div className="aspect-[3/4] rounded-2xl overflow-hidden shadow-xl mb-8">
                    <img
                      src={profile.profileImage}
                      alt={profile.name}
                      className="w-full h-full object-cover"
                    />
                  </div>
                )}

                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-heading font-bold text-neutral-900 mb-4">Connect</h3>
                    <div className="space-y-4">
                      <a href={`mailto:${profile.email}`} className="flex items-center text-neutral-600 hover:text-blue-600 transition-colors group">
                        <div className="w-10 h-10 rounded-full bg-neutral-50 flex items-center justify-center mr-4 group-hover:bg-blue-50 transition-colors">
                          <Mail size={18} />
                        </div>
                        <span className="font-medium">{profile.email}</span>
                      </a>

                      <a href={profile.social.linkedin} target="_blank" rel="noopener noreferrer" className="flex items-center text-neutral-600 hover:text-blue-600 transition-colors group">
                        <div className="w-10 h-10 rounded-full bg-neutral-50 flex items-center justify-center mr-4 group-hover:bg-blue-50 transition-colors">
                          <Linkedin size={18} />
                        </div>
                        <span className="font-medium">LinkedIn</span>
                      </a>

                      <a href={profile.social.github} target="_blank" rel="noopener noreferrer" className="flex items-center text-neutral-600 hover:text-blue-600 transition-colors group">
                        <div className="w-10 h-10 rounded-full bg-neutral-50 flex items-center justify-center mr-4 group-hover:bg-blue-50 transition-colors">
                          <Github size={18} />
                        </div>
                        <span className="font-medium">GitHub</span>
                      </a>

                      <a href={profile.social.twitter} target="_blank" rel="noopener noreferrer" className="flex items-center text-neutral-600 hover:text-blue-600 transition-colors group">
                        <div className="w-10 h-10 rounded-full bg-neutral-50 flex items-center justify-center mr-4 group-hover:bg-blue-50 transition-colors">
                          {/* Modern X Icon */}
                          <svg viewBox="0 0 24 24" fill="currentColor" className="w-[18px] h-[18px]">
                            <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                          </svg>
                        </div>
                        <span className="font-medium">Twitter / X</span>
                      </a>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-lg font-heading font-bold text-neutral-900 mb-2">Location</h3>
                    <div className="flex items-center text-neutral-600">
                      <MapPin size={18} className="mr-3" /> {profile.location}
                    </div>
                  </div>
                </div>
              </div>
            </Reveal>
          </div>
        </div>
      </div>

      {/* Awards Section - Full Width Background Wrapper */}
      {profile.achievements.length > 0 && (
        <section className="bg-neutral-50 pt-16 pb-24 mb-24 border-t border-neutral-100">
          <div className="container mx-auto px-6 max-w-6xl">
            <Reveal delay={0.3} width="100%">
              <div>
                <h2 className="font-heading font-bold text-3xl md:text-4xl text-neutral-900 mb-12">
                  Awards & Recognition
                </h2>

                {/* Slider Unit - Wider & Centered Container */}
                <div className="w-full flex flex-col items-center">
                  <div className="w-full max-w-5xl relative">
                    {/* Wide Slider Frame - Fixed aspect ratio, using object-contain to show full images */}
                    <div className="relative aspect-video rounded-[2.5rem] overflow-hidden shadow-2xl bg-neutral-50 border border-neutral-100">
                      {profile.achievements.map((achievement, index) => (
                        <div
                          key={index}
                          className={`absolute inset-0 transition-opacity duration-1000 ease-in-out flex items-center justify-center p-4 md:p-8 ${index === currentSlide ? 'opacity-100' : 'opacity-0'
                            }`}
                        >
                          <img
                            src={achievement.url}
                            alt={achievement.caption}
                            className="max-w-full max-h-full object-contain rounded-xl shadow-lg"
                          />
                        </div>
                      ))}
                    </div>

                    {/* Caption & Indicator Area */}
                    <div className="text-center mt-12">
                      <h4 className="text-xl md:text-2xl font-bold text-neutral-800 tracking-wider uppercase">
                        {profile.achievements[currentSlide]?.caption}
                      </h4>
                      <div className="w-20 h-1.5 bg-blue-600 mx-auto mt-5 rounded-full shadow-[0_2px_8px_rgba(37,99,235,0.4)]"></div>
                    </div>

                    {/* Navigation Dots */}
                    <div className="flex justify-center space-x-3 mt-10">
                      {profile.achievements.map((_, index) => (
                        <button
                          key={index}
                          onClick={() => setCurrentSlide(index)}
                          className={`h-2.5 rounded-full transition-all duration-500 ${index === currentSlide
                            ? 'bg-blue-600 w-12 shadow-sm'
                            : 'bg-neutral-300 w-2.5 hover:bg-neutral-400'
                            }`}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </Reveal>
          </div>
        </section>
      )}
    </div>
  );
};
