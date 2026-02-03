import React, { useState, useEffect, useRef } from 'react';
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
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Direct DOM event listener for close button
  useEffect(() => {
    const closeButton = closeButtonRef.current;
    if (!closeButton) return;

    const handleClose = (e: Event) => {
      e.preventDefault();
      e.stopPropagation();
      setIsOpen(false);
    };

    // Add multiple event types for maximum compatibility
    closeButton.addEventListener('touchstart', handleClose, { passive: false });
    closeButton.addEventListener('mousedown', handleClose);
    closeButton.addEventListener('click', handleClose);

    return () => {
      closeButton.removeEventListener('touchstart', handleClose);
      closeButton.removeEventListener('mousedown', handleClose);
      closeButton.removeEventListener('click', handleClose);
    };
  }, [isOpen]);

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

  // Lock body scroll when menu is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  return (
    <>
      <nav
        className={`fixed top-0 left-0 w-full z-50 transition-all duration-300 ${scrolled ? 'bg-white/90 backdrop-blur-md shadow-sm py-4' : 'bg-transparent py-6'
          }`}
      >
        <div className="container mx-auto px-6 flex justify-between items-center relative">
          <Link to="/">
            <img
              src="/images/logo-transparent.png"
              alt="Shrikant Naidu"
              className="h-[80px] w-auto"
            />
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
              type="button"
              className="md:hidden text-neutral-900 focus:outline-none p-2 -mr-2"
              onClick={() => setIsOpen(true)}
              aria-label="Open menu"
            >
              <Menu size={28} />
            </button>
          )}
        </div>
      </nav>

      {/* Mobile Menu Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 z-[9999] md:hidden bg-white"
          style={{ touchAction: 'none' }}
        >
          {/* Header with close button */}
          <div className="flex justify-between items-center p-6">
            <span className="text-xl font-heading font-bold text-neutral-900 tracking-tight">
              Menu
            </span>
            {/* Close Button - using ref for direct DOM event handling */}
            <button
              ref={closeButtonRef}
              type="button"
              className="w-14 h-14 flex items-center justify-center text-neutral-900 bg-neutral-100 rounded-full cursor-pointer select-none"
              style={{
                WebkitTapHighlightColor: 'transparent',
                touchAction: 'manipulation',
                userSelect: 'none'
              }}
              aria-label="Close menu"
            >
              <X size={28} strokeWidth={2.5} />
            </button>
          </div>

          {/* Navigation Links */}
          <div className="flex-grow flex flex-col items-center justify-center h-[calc(100%-200px)]">
            <div className="flex flex-col space-y-6 text-center px-6 w-full">
              {navItems.map((item) => (
                <a
                  key={item.label}
                  href={item.href}
                  className="text-3xl font-menu font-bold text-neutral-900 uppercase tracking-widest block py-2"
                  onClick={(e) => handleNavClick(e, item.href)}
                >
                  {item.label}
                </a>
              ))}
            </div>
          </div>

          {/* Footer */}
          <div className="absolute bottom-0 left-0 right-0 p-8 text-center border-t border-neutral-100">
            <div className="text-neutral-400 text-xs font-medium tracking-[0.2em] uppercase">
              © {new Date().getFullYear()} Shrikant Naidu
            </div>
          </div>
        </div>
      )}
    </>
  );
};