import React from 'react';
import { Hero } from '../components/Hero';
import { Works } from '../components/Works';
import { Testimonials } from '../components/Testimonials';
import { Writing } from '../components/Writing';
import { About } from '../components/About';
import { Contact } from '../components/Contact';

export const Home: React.FC = () => {
  return (
    <>
      <Hero />
      <Works />
      <Writing />
      <About />
      {/* <Testimonials /> */}
      <Contact />
    </>
  );
};