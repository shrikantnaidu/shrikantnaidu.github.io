import React from 'react';
import { Testimonial } from '../types';
import { Reveal } from './Reveal';
import { testimonials } from '../data';

export const Testimonials: React.FC = () => {
  return (
    <section className="py-24 bg-white relative">
      <div className="container mx-auto px-6 max-w-4xl">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
          {testimonials.map((t, index) => (
            <Reveal key={t.id} delay={index * 0.1}>
              <div className="flex flex-col items-start">
                <span className="text-4xl text-neutral-300 font-serif mb-6">“</span>
                <p className="text-lg text-neutral-600 leading-relaxed mb-6 font-medium">
                  {t.quote}
                </p>
                <div>
                  <cite className="not-italic font-bold text-neutral-900 block font-heading">
                    {t.author}
                  </cite>
                  <span className="text-sm text-neutral-500">{t.role}</span>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
};