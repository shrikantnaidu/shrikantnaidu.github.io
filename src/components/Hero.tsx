import React, { useState, useEffect, useRef } from 'react';
import { ArrowDown } from 'lucide-react';
import { Reveal } from './Reveal';
import { profile } from '../data';

export const Hero: React.FC = () => {
  const [textIndex, setTextIndex] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Text rotation logic
  useEffect(() => {
    const interval = setInterval(() => {
      setTextIndex((prev) => (prev + 1) % profile.hero.rotatingTexts.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  // Canvas Particle & Neural Impulse Animation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;

    type Particle = { x: number; y: number; vx: number; vy: number; size: number };
    type Impulse = {
      source: Particle;
      target: Particle;
      progress: number;
      speed: number;
    };

    let particles: Particle[] = [];
    let impulses: Impulse[] = [];

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      initParticles();
    };

    const initParticles = () => {
      particles = [];
      impulses = [];

      const particleCount = Math.min(Math.floor(window.innerWidth / 15), 80);

      for (let i = 0; i < particleCount; i++) {
        particles.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: (Math.random() - 0.5) * 0.3,
          vy: (Math.random() - 0.5) * 0.3,
          size: Math.random() * 1.5 + 0.5,
        });
      }
    };

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // --- BACKGROUND NETWORK ---
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = '#f3f4f6';

      particles.forEach((p, i) => {
        p.x += p.vx;
        p.y += p.vy;

        // Bounce off edges
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        // Draw Node
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = '#9ca3af';
        ctx.fill();

        // Draw Connections
        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const dx = p.x - p2.x;
          const dy = p.y - p2.y;
          const distSq = dx * dx + dy * dy;
          const maxDist = 20000;

          if (distSq < maxDist) {
            ctx.beginPath();
            ctx.globalAlpha = (1 - Math.sqrt(distSq) / Math.sqrt(maxDist)) * 0.2;
            ctx.strokeStyle = '#9ca3af';
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
            ctx.globalAlpha = 1.0;

            // Chance to spawn an electric impulse
            if (Math.random() < 0.0005) {
              impulses.push({
                source: p,
                target: p2,
                progress: 0,
                speed: 0.005 + Math.random() * 0.01
              });
            }
          }
        }
      });

      // --- ELECTRIC IMPULSES ---
      // Optimized Loop: remove filter() allocation to reduce GC
      ctx.globalCompositeOperation = 'lighter';

      let writeIdx = 0; // In-place array compaction
      for (let i = 0; i < impulses.length; i++) {
        const imp = impulses[i];
        imp.progress += imp.speed;

        if (imp.progress < 1) {
          const dx = imp.target.x - imp.source.x;
          const dy = imp.target.y - imp.source.y;
          const distSq = dx * dx + dy * dy;

          // Only keep if connection valid
          if (distSq < 25000) {
            const currX = imp.source.x + dx * imp.progress;
            const currY = imp.source.y + dy * imp.progress;

            ctx.beginPath();
            ctx.arc(currX, currY, 2, 0, Math.PI * 2);
            ctx.fillStyle = '#3b82f6'; // Blue-500
            ctx.shadowColor = '#2563eb';
            ctx.shadowBlur = 8;
            ctx.fill();
            ctx.shadowBlur = 0;

            // Keep in array
            impulses[writeIdx++] = imp;
          }
        }
      }
      impulses.length = writeIdx; // Truncate array

      ctx.globalCompositeOperation = 'source-over';

      animationFrameId = requestAnimationFrame(draw);
    };

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    draw();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  const handleScroll = (e: React.MouseEvent<HTMLAnchorElement>) => {
    e.preventDefault();
    const element = document.querySelector('#works');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const HeroContent = () => (
    <div
      className="flex flex-col items-center text-center max-w-5xl mx-auto px-6 py-20"
    >
      <h1 className="text-6xl md:text-8xl lg:text-9xl font-heading font-black text-neutral-900 leading-[0.9] mb-8 tracking-tighter text-center uppercase">
        {profile.hero.heading} <br />
        <span className="text-blue-600 block">
          <span key={textIndex} className="block animate-fade-in-up">
            {profile.hero.rotatingTexts[textIndex]}
          </span>
        </span>
      </h1>

      <p className="text-xl md:text-2xl text-neutral-500 max-w-3xl leading-relaxed mb-12 font-medium text-center mx-auto uppercase tracking-wide">
        {profile.hero.subheading}
      </p>

      <a
        href="#works"
        onClick={handleScroll}
        className="group inline-flex items-center gap-2 px-8 py-4 bg-neutral-900 text-white rounded-none font-bold hover:bg-neutral-800 transition-all hover:scale-105 hover:shadow-xl uppercase tracking-widest text-sm border-2 border-transparent hover:border-blue-600 pointer-events-auto"
      >
        "Explore"
        <ArrowDown
          size={20}
          className="group-hover:translate-y-1 transition-transform duration-300"
        />
      </a>
    </div>
  );

  return (
    <section id="home" className="min-h-screen flex items-center justify-center relative overflow-hidden bg-white">
      {/* Background Canvas */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none z-0"
      />

      {/* Grid Overlay */}
      <div
        className="absolute inset-0 w-full h-full pointer-events-none opacity-[0.05] z-0"
        style={{
          backgroundImage: `linear-gradient(#000 1px, transparent 1px), linear-gradient(90deg, #000 1px, transparent 1px)`,
          backgroundSize: '40px 40px'
        }}
      ></div>

      {/* Main Content */}
      <Reveal width="100%">
        <div className="relative w-full z-10 min-h-[80vh] flex items-center justify-center">
          <HeroContent />
        </div>
      </Reveal>
    </section>
  );
};
