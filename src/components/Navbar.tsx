import React, { useState, useEffect } from 'react';
import { Menu, X } from 'lucide-react';
import { NavItem } from '../types';
import { Link, useLocation, useNavigate } from 'react-router-dom';

const navItems: NavItem[] = [
  { label: 'Home', href: '#home' },
  { label: 'Works', href: '#works' },
  { label: 'Writing', href: '#writing' },
  { label: 'About', href: '#about' },
  { label: 'Contact', href: '#contact' },
];

export const Navbar: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, href: string) => {
    e.preventDefault();
    setIsOpen(false);

    const targetId = href;

    if (location.pathname !== '/') {
      navigate('/');
      setTimeout(() => {
        const element = document.querySelector(targetId);
        element?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } else {
      const element = document.querySelector(targetId);
      element?.scrollIntoView({ behavior: 'smooth' });
    }
  };

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);

  return (
    <>
      <nav
        className={`fixed top-0 left-0 w-full z-50 transition-all duration-300 ${scrolled ? 'bg-white/90 backdrop-blur-md shadow-sm py-4' : 'bg-transparent py-6'
          }`}
      >
        <div className="container mx-auto px-6 flex justify-between items-center relative">
          <Link to="/" className="text-xl font-heading font-bold text-neutral-900 tracking-tight">
            Shrikant Naidu
          </Link>

          {/* Desktop Menu */}
          <div className="hidden md:flex items-center space-x-8">
            {navItems.map((item) => (
              <a
                key={item.label}
                href={item.href}
                onClick={(e) => handleNavClick(e, item.href)}
                className="text-sm font-menu font-medium text-neutral-500 hover:text-neutral-900 transition-colors duration-200 uppercase tracking-widest"
              >
                {item.label}
              </a>
            ))}
          </div>

          {/* Mobile Toggle */}
          {!isOpen && (
            <button
              className="md:hidden text-neutral-900 focus:outline-none p-2 -mr-2"
              onClick={() => setIsOpen(true)}
              aria-label="Open menu"
            >
              <Menu size={28} />
            </button>
          )}
        </div>
      </nav>

      <div
        className={`fixed inset-0 bg-white z-[100] flex flex-col transition-all duration-500 ease-in-out ${isOpen ? 'opacity-100 translate-y-0 pointer-events-auto' : 'opacity-0 -translate-y-full pointer-events-none'
          }`}
      >
        {/* Mobile Overlay Header */}
        <div className="flex justify-between items-center p-6 h-24">
          <span className="text-xl font-heading font-bold text-neutral-900 tracking-tight">
            Menu
          </span>
          <button
            className="text-neutral-900 focus:outline-none p-4 hover:bg-neutral-50 rounded-full transition-all active:scale-90 flex items-center justify-center bg-white shadow-sm border border-neutral-100"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setIsOpen(false);
            }}
            aria-label="Close menu"
          >
            <X size={36} />
          </button>
        </div>

        <div className="flex-grow flex flex-col items-center justify-center -mt-20">
          <div className="flex flex-col space-y-8 text-center px-6 w-full">
            {navItems.map((item, index) => (
              <a
                key={item.label}
                href={item.href}
                className={`text-4xl font-menu font-bold text-neutral-900 hover:text-blue-600 transition-all duration-500 uppercase tracking-widest block py-3 border-b border-transparent ${isOpen ? 'opacity-100 translate-y-0 scale-100' : 'opacity-0 translate-y-8 scale-95'
                  }`}
                style={{ transitionDelay: `${index * 75}ms` }}
                onClick={(e) => handleNavClick(e, item.href)}
              >
                {item.label}
              </a>
            ))}
          </div>
        </div>

        <div className="p-8 text-center border-t border-neutral-50">
          <div className="text-neutral-400 text-xs font-medium tracking-[0.2em] uppercase mb-2">
            © {new Date().getFullYear()} Shrikant Naidu
          </div>
        </div>
      </div>
    </>
  );
};