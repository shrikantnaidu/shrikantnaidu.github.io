import React, { useState } from 'react';
import { Reveal } from './Reveal';
import { profile } from '../data';
import { CheckCircle2, AlertCircle, Send, Loader2 } from 'lucide-react';

export const Contact: React.FC = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: ''
  });
  const [status, setStatus] = useState<'idle' | 'submitting' | 'success' | 'error'>('idle');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('submitting');

    // Formspree endpoint handling (supports both ID and full URL)
    const idOrUrl = profile.contact.formspreeId || 'your-formspree-id';
    const endpoint = idOrUrl.startsWith('http') ? idOrUrl : `https://formspree.io/f/${idOrUrl}`;

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(formData)
      });

      if (response.ok) {
        setStatus('success');
        setFormData({ name: '', email: '', message: '' });
      } else {
        setStatus('error');
      }
    } catch (error) {
      console.error('Form submission error:', error);
      setStatus('error');
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { id, value } = e.target;
    setFormData(prev => ({ ...prev, [id]: value }));
  };

  return (
    <section id="contact" className="py-16 md:py-24 bg-neutral-50 scroll-mt-24">
      <div className="container mx-auto px-6 max-w-3xl">
        <Reveal>
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-heading font-bold text-neutral-900 mb-4">
              Get in touch
            </h2>
            <p className="text-neutral-600 text-lg md:text-xl">
              Interested in collaboration? Let's discuss your data strategy.
            </p>
          </div>
        </Reveal>

        <Reveal width="100%" delay={0.2}>
          {status === 'success' ? (
            <div className="text-center py-12 bg-white rounded-2xl shadow-sm border border-neutral-100 animate-fade-in">
              <div className="w-16 h-16 bg-green-50 text-green-600 rounded-full flex items-center justify-center mx-auto mb-6">
                <CheckCircle2 size={32} />
              </div>
              <h3 className="text-2xl font-bold text-neutral-900 mb-2">Message Sent!</h3>
              <p className="text-neutral-500 mb-8 px-6">Thanks for reaching out. I'll get back to you soon.</p>
              <button
                onClick={() => setStatus('idle')}
                className="px-8 py-3 bg-neutral-900 text-white font-medium rounded-lg hover:bg-neutral-800 transition-all shadow-md hover:shadow-lg"
              >
                Send Another
              </button>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <label htmlFor="name" className="text-base font-medium text-neutral-700 ml-1">Name</label>
                  <input
                    required
                    type="text"
                    id="name"
                    value={formData.name}
                    onChange={handleChange}
                    className="w-full px-4 py-3 rounded-lg bg-white border border-neutral-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all text-neutral-900 shadow-sm"
                    placeholder="John Doe"
                  />
                </div>
                <div className="space-y-2">
                  <label htmlFor="email" className="text-base font-medium text-neutral-700 ml-1">Email</label>
                  <input
                    required
                    type="email"
                    id="email"
                    value={formData.email}
                    onChange={handleChange}
                    className="w-full px-4 py-3 rounded-lg bg-white border border-neutral-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all text-neutral-900 shadow-sm"
                    placeholder="john@example.com"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <label htmlFor="message" className="text-base font-medium text-neutral-700 ml-1">Message</label>
                <textarea
                  required
                  id="message"
                  rows={4}
                  value={formData.message}
                  onChange={handleChange}
                  className="w-full px-4 py-3 rounded-lg bg-white border border-neutral-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition-all resize-none text-neutral-900 shadow-sm"
                  placeholder="How can I help you?"
                ></textarea>
              </div>

              {status === 'error' && (
                <div className="flex items-center gap-3 p-4 bg-red-50 text-red-600 rounded-lg">
                  <AlertCircle size={20} />
                  <p className="text-sm font-medium">Something went wrong. Please try again.</p>
                </div>
              )}

              <button
                disabled={status === 'submitting'}
                type="submit"
                className="w-full py-4 bg-neutral-900 text-white font-medium rounded-lg hover:bg-neutral-800 transition-all shadow-md hover:shadow-lg transform hover:-translate-y-0.5 flex items-center justify-center gap-2 disabled:opacity-70 disabled:cursor-not-allowed"
              >
                {status === 'submitting' ? (
                  <>
                    <Loader2 className="animate-spin" size={20} />
                    Sending...
                  </>
                ) : (
                  <>
                    Send Message
                    <Send size={18} />
                  </>
                )}
              </button>
            </form>
          )}
        </Reveal>
      </div>
    </section>
  );
};