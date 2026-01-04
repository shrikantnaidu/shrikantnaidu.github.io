import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, MapPin, Mail, Linkedin, Github, ChevronLeft, ChevronRight } from 'lucide-react';
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

                      {profile.social.linkedin && (
                        <a href={profile.social.linkedin} target="_blank" rel="noopener noreferrer" className="flex items-center text-neutral-600 hover:text-blue-600 transition-colors group">
                          <div className="w-10 h-10 rounded-full bg-neutral-50 flex items-center justify-center mr-4 group-hover:bg-blue-50 transition-colors text-neutral-500 group-hover:text-blue-600">
                            <Linkedin size={18} />
                          </div>
                          <span className="font-medium">LinkedIn</span>
                        </a>
                      )}

                      {profile.social.github && (
                        <a href={profile.social.github} target="_blank" rel="noopener noreferrer" className="flex items-center text-neutral-600 hover:text-blue-600 transition-colors group">
                          <div className="w-10 h-10 rounded-full bg-neutral-50 flex items-center justify-center mr-4 group-hover:bg-blue-50 transition-colors text-neutral-500 group-hover:text-blue-600">
                            <Github size={18} />
                          </div>
                          <span className="font-medium">GitHub</span>
                        </a>
                      )}

                      {profile.social.twitter && (
                        <a href={profile.social.twitter} target="_blank" rel="noopener noreferrer" className="flex items-center text-neutral-600 hover:text-blue-600 transition-colors group">
                          <div className="w-10 h-10 rounded-full bg-neutral-50 flex items-center justify-center mr-4 group-hover:bg-blue-50 transition-colors text-neutral-500 group-hover:text-blue-600">
                            <svg viewBox="0 0 24 24" fill="currentColor" className="w-[18px] h-[18px]">
                              <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                            </svg>
                          </div>
                          <span className="font-medium">Twitter / X</span>
                        </a>
                      )}

                      {profile.social.lightning && (
                        <a href={profile.social.lightning} target="_blank" rel="noopener noreferrer" className="flex items-center text-neutral-600 hover:text-blue-600 transition-colors group">
                          <div className="w-10 h-10 rounded-full bg-neutral-50 flex items-center justify-center mr-4 group-hover:bg-blue-50 transition-colors text-neutral-500 group-hover:text-blue-600">
                            <svg viewBox="0 0 33 38" fill="currentColor" className="w-[18px] h-[18px]">
                              <path d="M15.5629 0.195181L0.711016 8.96731C0.494828 9.09534 0.315359 9.2791 0.190577 9.50023C0.0657942 9.72135 8.06666e-05 9.97205 0 10.2273V27.7739C8.06666e-05 28.0291 0.0657942 28.2798 0.190577 28.501C0.315359 28.7221 0.494828 28.9058 0.711016 29.0339L15.5629 37.8048C15.7798 37.9326 16.0259 38 16.2763 38C16.5267 38 16.7728 37.9326 16.9896 37.8048L31.8416 29.0339C32.0577 28.9058 32.2372 28.7221 32.362 28.501C32.4868 28.2798 32.5525 28.0291 32.5526 27.7739V10.2273C32.5525 9.97205 32.4868 9.72135 32.362 9.50023C32.2372 9.2791 32.0577 9.09534 31.8416 8.96731L16.9896 0.195181C16.7728 0.0672991 16.5267 0 16.2763 0C16.0259 0 15.7798 0.0672991 15.5629 0.195181V0.195181ZM13.1422 29.4447L14.9057 21.7726C14.9183 21.717 14.9168 21.659 14.9013 21.6043C14.8857 21.5495 14.8566 21.4997 14.8168 21.4596L10.5343 17.202C10.5037 17.1712 10.4793 17.1345 10.4627 17.094C10.4461 17.0535 10.4375 17.01 10.4375 16.9661C10.4375 16.9222 10.4461 16.8787 10.4627 16.8382C10.4793 16.7977 10.5037 16.761 10.5343 16.7302L18.8923 8.24117C18.942 8.19024 19.0066 8.15729 19.0762 8.14731C19.1459 8.13734 19.2169 8.15089 19.2784 8.18592C19.3398 8.22094 19.3884 8.27551 19.4166 8.34133C19.4449 8.40714 19.4513 8.48061 19.435 8.5505L17.6691 16.2418C17.6559 16.2973 17.6571 16.3554 17.6727 16.4103C17.6883 16.4652 17.7177 16.515 17.758 16.5547L22.0077 20.7897C22.0384 20.8203 22.0627 20.857 22.0793 20.8974C22.0959 20.9378 22.1044 20.9811 22.1044 21.0249C22.1044 21.0687 22.0959 21.1121 22.0793 21.1525C22.0627 21.1929 22.0384 21.2295 22.0077 21.2602L13.6778 29.7552C13.6277 29.8048 13.5632 29.8366 13.494 29.8458C13.4248 29.855 13.3545 29.8411 13.2937 29.8062C13.2328 29.7713 13.1846 29.7172 13.1563 29.6521C13.1279 29.587 13.1209 29.5142 13.1363 29.4447H13.1422Z" />
                            </svg>
                          </div>
                          <span className="font-medium">Lightning AI</span>
                        </a>
                      )}

                      {profile.social.wandb && (
                        <a href={profile.social.wandb} target="_blank" rel="noopener noreferrer" className="flex items-center text-neutral-600 hover:text-blue-600 transition-colors group">
                          <div className="w-10 h-10 rounded-full bg-neutral-50 flex items-center justify-center mr-4 group-hover:bg-blue-50 transition-colors text-neutral-500 group-hover:text-blue-600">
                            <svg viewBox="0 0 170 170" fill="currentColor" className="w-[18px] h-[18px]">
                              <g>
                                <path d="M17.259 21.629C23.249 21.629 28.041 16.857 28.041 10.8924C28.041 4.9279 23.249 0.156189 17.259 0.156189C11.2687 0.156189 6.47656 4.9279 6.47656 10.8924C6.47656 16.857 11.2687 21.629 17.259 21.629Z" />
                                <path d="M21.063 70.155C30.248 67.968 36.038 58.822 33.842 49.676C31.646 40.531 22.461 34.765 13.276 36.952C4.09097 39.139 -1.69943 48.285 0.496972 57.43C2.69327 66.775 11.8782 72.342 21.063 70.155Z" />
                                <path d="M21.063 155.448C30.248 153.26 36.038 144.115 33.842 134.969C31.646 125.824 22.461 120.058 13.276 122.245C4.09097 124.432 -1.69943 133.578 0.496972 142.723C2.69327 151.869 11.8782 157.635 21.063 155.448Z" />
                                <path d="M27.869 96.185C27.869 90.221 23.077 85.449 17.087 85.449C11.0968 85.449 6.30469 90.221 6.30469 96.185C6.30469 102.15 11.0968 106.922 17.087 106.922C23.077 106.922 27.869 102.15 27.869 96.185Z" />
                                <path d="M82.9769 95C73.5919 95 65.8049 102.555 65.8049 112.098C65.8049 121.443 73.3919 129.197 82.9769 129.197C92.3609 129.197 100.148 121.641 100.148 112.098C99.9479 102.555 92.3609 95 82.9769 95Z" />
                                <path d="M82.9781 143.902C76.9881 143.902 72.1951 148.674 72.1951 154.639C72.1951 160.603 76.9881 165.375 82.9781 165.375C88.9681 165.375 93.7601 160.603 93.7601 154.639C93.5601 148.674 88.7681 143.902 82.9781 143.902Z" />
                                <path d="M82.9781 80.082C88.9681 80.082 93.7601 75.31 93.7601 69.345C93.7601 63.381 88.9681 58.609 82.9781 58.609C76.9881 58.609 72.1951 63.381 72.1951 69.345C72.1951 75.31 76.9881 80.082 82.9781 80.082Z" />
                                <path d="M84.5601 37.345C90.3501 36.351 94.3431 30.983 93.3451 25.217C92.3471 19.451 86.9561 15.475 81.1651 16.469C75.3751 17.463 71.3811 22.831 72.3801 28.597C73.3781 34.363 78.7691 38.14 84.5601 37.345Z" />
                                <path d="M160.649 65.579C167.238 59.018 167.238 48.083 160.649 41.522C154.06 34.961 143.078 34.961 136.489 41.522C129.9 48.083 129.9 59.018 136.489 65.579C143.078 72.339 154.06 72.339 160.649 65.579Z" />
                                <path d="M148.665 21.629C154.655 21.629 159.447 16.857 159.447 10.8924C159.447 4.9279 154.655 0.156189 148.665 0.156189C142.675 0.156189 137.883 4.9279 137.883 10.8924C137.883 16.857 142.675 21.629 148.665 21.629Z" />
                                <path d="M148.665 85.648C142.675 85.648 137.883 90.42 137.883 96.385C137.883 102.349 142.675 107.121 148.665 107.121C154.655 107.121 159.447 102.349 159.447 96.385C159.248 90.42 154.455 85.648 148.665 85.648Z" />
                                <path d="M148.665 128.203C142.675 128.203 137.883 132.975 137.883 138.939C137.883 144.904 142.675 149.675 148.665 149.675C154.655 149.675 159.447 144.904 159.447 138.939C159.248 132.975 154.455 128.203 148.665 128.203Z" />
                              </g>
                            </svg>
                          </div>
                          <span className="font-medium">Weights & Biases</span>
                        </a>
                      )}

                      {profile.social.huggingface && (
                        <a href={profile.social.huggingface} target="_blank" rel="noopener noreferrer" className="flex items-center text-neutral-600 hover:text-blue-600 transition-colors group">
                          <div className="w-10 h-10 rounded-full bg-neutral-50 flex items-center justify-center mr-4 group-hover:bg-blue-50 transition-colors text-neutral-500 group-hover:text-blue-600">
                            <svg viewBox="0 0 24 24" fill="currentColor" className="w-[18px] h-[18px]">
                              <path d="M12.025 1.13c-5.77 0-10.449 4.647-10.449 10.378 0 1.112.178 2.181.503 3.185.064-.222.203-.444.416-.577a.96.96 0 0 1 .524-.15c.293 0 .584.124.84.284.278.173.48.408.71.694.226.282.458.611.684.951v-.014c.017-.324.106-.622.264-.874s.403-.487.762-.543c.3-.047.596.06.787.203s.31.313.4.467c.15.257.212.468.233.542.01.026.653 1.552 1.657 2.54.616.605 1.01 1.223 1.082 1.912.055.537-.096 1.059-.38 1.572.637.121 1.294.187 1.967.187.657 0 1.298-.063 1.921-.178-.287-.517-.44-1.041-.384-1.581.07-.69.465-1.307 1.081-1.913 1.004-.987 1.647-2.513 1.657-2.539.021-.074.083-.285.233-.542.09-.154.208-.323.4-.467a1.08 1.08 0 0 1 .787-.203c.359.056.604.29.762.543s.247.55.265.874v.015c.225-.34.457-.67.683-.952.23-.286.432-.52.71-.694.257-.16.547-.284.84-.285a.97.97 0 0 1 .524.151c.228.143.373.388.43.625l.006.04a10.3 10.3 0 0 0 .534-3.273c0-5.731-4.678-10.378-10.449-10.378M8.327 6.583a1.5 1.5 0 0 1 .713.174 1.487 1.487 0 0 1 .617 2.013c-.183.343-.762-.214-1.102-.094-.38.134-.532.914-.917.71a1.487 1.487 0 0 1 .69-2.803m7.486 0a1.487 1.487 0 0 1 .689 2.803c-.385.204-.536-.576-.916-.71-.34-.12-.92.437-1.103.094a1.487 1.487 0 0 1 .617-2.013 1.5 1.5 0 0 1 .713-.174m-10.68 1.55a.96.96 0 1 1 0 1.921.96.96 0 0 1 0-1.92m13.838 0a.96.96 0 1 1 0 1.92.96.96 0 0 1 0-1.92M8.489 11.458c.588.01 1.965 1.157 3.572 1.164 1.607-.007 2.984-1.155 3.572-1.164.196-.003.305.12.305.454 0 .886-.424 2.328-1.563 3.202-.22-.756-1.396-1.366-1.63-1.32q-.011.001-.02.006l-.044.026-.01.008-.03.024q-.018.017-.035.036l-.032.04a1 1 0 0 0-.058.09l-.014.025q-.049.088-.11.19a1 1 0 0 1-.083.116 1.2 1.2 0 0 1-.173.18q-.035.029-.075.058a1.3 1.3 0 0 1-.251-.243 1 1 0 0 1-.076-.107c-.124-.193-.177-.363-.337-.444-.034-.016-.104-.008-.2.022q-.094.03-.216.087-.06.028-.125.063l-.13.074q-.067.04-.136.086a3 3 0 0 0-.135.096 3 3 0 0 0-.26.219 2 2 0 0 0-.12.121 2 2 0 0 0-.106.128l-.002.002a2 2 0 0 0-.09.132l-.001.001a1.2 1.2 0 0 0-.105.212q-.013.036-.024.073c-1.139-.875-1.563-2.317-1.563-3.203 0-.334.109-.457.305-.454m.836 10.354c.824-1.19.766-2.082-.365-3.194-1.13-1.112-1.789-2.738-1.789-2.738s-.246-.945-.806-.858-.97 1.499.202 2.362c1.173.864-.233 1.45-.685.64-.45-.812-1.683-2.896-2.322-3.295s-1.089-.175-.938.647 2.822 2.813 2.562 3.244-1.176-.506-1.176-.506-2.866-2.567-3.49-1.898.473 1.23 2.037 2.16c1.564.932 1.686 1.178 1.464 1.53s-3.675-2.511-4-1.297c-.323 1.214 3.524 1.567 3.287 2.405-.238.839-2.71-1.587-3.216-.642-.506.946 3.49 2.056 3.522 2.064 1.29.33 4.568 1.028 5.713-.624m5.349 0c-.824-1.19-.766-2.082.365-3.194 1.13-1.112 1.789-2.738 1.789-2.738s-.246-.945.806-.858.97 1.499-.202 2.362c-1.173.864.233 1.45.685.64.451-.812 1.683-2.896 2.322-3.295s1.089-.175.938.647-2.822 2.813-2.562 3.244 1.176-.506 1.176-.506 2.866-2.567 3.49-1.898-.473 1.23-2.037 2.16c-1.564.932-1.686 1.178-1.464 1.53s3.675-2.511 4-1.297c.323 1.214-3.524 1.567-3.287 2.405.238.839 2.71-1.587 3.216-.642.506.946-3.49 2.056-3.522 2.064-1.29.33-4.568 1.028-5.713-.624" />
                            </svg>
                          </div>
                          <span className="font-medium">Hugging Face</span>
                        </a>
                      )}

                      {profile.social.steam && (
                        <a href={profile.social.steam} target="_blank" rel="noopener noreferrer" className="flex items-center text-neutral-600 hover:text-blue-600 transition-colors group">
                          <div className="w-10 h-10 rounded-full bg-neutral-50 flex items-center justify-center mr-4 group-hover:bg-blue-50 transition-colors text-neutral-500 group-hover:text-blue-600">
                            <svg viewBox="0 0 24 24" fill="currentColor" className="w-[18px] h-[18px]">
                              <path d="M11.979 0C5.678 0 .511 4.86.022 11.037l6.432 2.658c.545-.371 1.203-.59 1.912-.59.063 0 .125.004.188.006l2.861-4.142V8.91c0-2.495 2.028-4.524 4.524-4.524 2.494 0 4.524 2.031 4.524 4.527s-2.03 4.525-4.524 4.525h-.105l-4.076 2.911c0 .052.004.105.004.159 0 1.875-1.515 3.396-3.39 3.396-1.635 0-3.016-1.173-3.331-2.727L.436 15.27C1.862 20.307 6.486 24 11.979 24c6.627 0 11.999-5.373 11.999-12S18.605 0 11.979 0zM7.54 18.21l-1.473-.61c.262.543.714.999 1.314 1.25 1.297.539 2.793-.076 3.332-1.375.263-.63.264-1.319.005-1.949s-.75-1.121-1.377-1.383c-.624-.26-1.29-.249-1.878-.03l1.523.63c.956.4 1.409 1.5 1.009 2.455-.397.957-1.497 1.41-2.454 1.012H7.54zm11.415-9.303c0-1.662-1.353-3.015-3.015-3.015-1.665 0-3.015 1.353-3.015 3.015 0 1.665 1.35 3.015 3.015 3.015 1.663 0 3.015-1.35 3.015-3.015zm-5.273-.005c0-1.252 1.013-2.266 2.265-2.266 1.249 0 2.266 1.014 2.266 2.266 0 1.251-1.017 2.265-2.266 2.265-1.253 0-2.265-1.014-2.265-2.265z" />
                            </svg>
                          </div>
                          <span className="font-medium">Steam</span>
                        </a>
                      )}
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
                    {/* Wide Slider Frame */}
                    <div className="relative aspect-video rounded-[2.5rem] overflow-hidden shadow-2xl bg-neutral-50 border border-neutral-100">
                      {profile.achievements.map((achievement, index) => (
                        <div
                          key={index}
                          className={`absolute inset-0 transition-all duration-1000 ease-in-out flex items-center justify-center p-4 md:p-8 ${index === currentSlide ? 'opacity-100 scale-100' : 'opacity-0 scale-95 pointer-events-none'
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

                    {/* Caption Section */}
                    <div className="text-center mt-12">
                      <h4 className="text-xl md:text-2xl font-bold text-neutral-800 tracking-wider uppercase">
                        {profile.achievements[currentSlide]?.caption}
                      </h4>
                    </div>

                    {/* Navigation Dots & Arrows Group */}
                    <div className="flex items-center justify-center space-x-8 mt-10">
                      <button
                        onClick={() => setCurrentSlide((prev) => (prev - 1 + profile.achievements.length) % profile.achievements.length)}
                        className="p-3 rounded-full bg-white shadow-md text-neutral-400 hover:text-blue-600 hover:shadow-lg transition-all transform hover:-translate-x-1"
                        aria-label="Previous slide"
                      >
                        <ChevronLeft size={24} />
                      </button>

                      <div className="flex space-x-3">
                        {profile.achievements.map((_, index) => (
                          <button
                            key={index}
                            onClick={() => setCurrentSlide(index)}
                            className={`h-2.5 rounded-full transition-all duration-500 ${index === currentSlide
                              ? 'bg-blue-600 w-12 shadow-sm'
                              : 'bg-neutral-300 w-2.5 hover:bg-neutral-400'
                              }`}
                            aria-label={`Go to slide ${index + 1}`}
                          />
                        ))}
                      </div>

                      <button
                        onClick={() => setCurrentSlide((prev) => (prev + 1) % profile.achievements.length)}
                        className="p-3 rounded-full bg-white shadow-md text-neutral-400 hover:text-blue-600 hover:shadow-lg transition-all transform hover:translate-x-1"
                        aria-label="Next slide"
                      >
                        <ChevronRight size={24} />
                      </button>
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
