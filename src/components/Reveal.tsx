import React, { useEffect, useRef, useState } from 'react';

interface Props {
  children: React.ReactNode;
  width?: 'fit-content' | '100%';
  height?: 'auto' | '100%';
  delay?: number;
  className?: string;
}

export const Reveal: React.FC<Props> = ({
  children,
  width = 'fit-content',
  height = 'auto',
  delay = 0,
  className = ''
}) => {
  const ref = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className={className}
      style={{
        width,
        height,
        position: 'relative',
        overflow: 'hidden'
      }}
    >
      <div
        style={{
          height: height === '100%' ? '100%' : 'auto',
          transform: isVisible ? 'translateY(0)' : 'translateY(75px)',
          opacity: isVisible ? 1 : 0,
          transition: `all 0.5s cubic-bezier(0.17, 0.55, 0.55, 1) ${delay}s`,
        }}
      >
        {children}
      </div>
    </div>
  );
};