import React from 'react';
import { Reveal } from './Reveal';

export const Contact: React.FC = () => {
  return (
    <section id="contact" className="py-24 bg-neutral-50 scroll-mt-24">
      <div className="container mx-auto px-6 max-w-3xl">
        <Reveal>
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-heading font-bold text-neutral-900 mb-4">
              Get in touch
            </h2>
            <p className="text-neutral-500 text-lg">
              Interested in collaboration? Let's discuss your data strategy.
            </p>
          </div>
        </Reveal>

        <Reveal width="100%" delay={0.2}>
          <form className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <label htmlFor="name" className="text-sm font-medium text-neutral-700">Name</label>
                <input
                  type="text"
                  id="name"
                  className="w-full px-4 py-3 rounded-lg bg-white border border-neutral-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all"
                  placeholder="John Doe"
                />
              </div>
              <div className="space-y-2">
                <label htmlFor="email" className="text-sm font-medium text-neutral-700">Email</label>
                <input
                  type="email"
                  id="email"
                  className="w-full px-4 py-3 rounded-lg bg-white border border-neutral-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all"
                  placeholder="john@example.com"
                />
              </div>
            </div>
            
            <div className="space-y-2">
              <label htmlFor="message" className="text-sm font-medium text-neutral-700">Message</label>
              <textarea
                id="message"
                rows={4}
                className="w-full px-4 py-3 rounded-lg bg-white border border-neutral-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all resize-none"
                placeholder="How can I help you?"
              ></textarea>
            </div>

            <button
              type="submit"
              className="w-full py-4 bg-neutral-900 text-white font-medium rounded-lg hover:bg-neutral-800 transition-all shadow-md hover:shadow-lg transform hover:-translate-y-0.5"
            >
              Send Message
            </button>
          </form>
        </Reveal>
      </div>
    </section>
  );
};