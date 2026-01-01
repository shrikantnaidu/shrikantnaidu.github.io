import React from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { GoogleAnalytics } from './components/GoogleAnalytics';
import { Home } from './pages/Home';
import { WorksPage } from './pages/WorksPage';
import { ProjectPage } from './pages/ProjectPage';
import { WritingPage } from './pages/WritingPage';
import { PostPage } from './pages/PostPage';
import { AboutPage } from './pages/AboutPage';

const App: React.FC = () => {
  const gaId = import.meta.env.VITE_GA_MEASUREMENT_ID || 'G-XXXXXXXXXX';

  return (
    <Router>
      <GoogleAnalytics measurementId={gaId} />
      <div className="antialiased text-neutral-900 bg-white min-h-screen relative font-sans">
        <Navbar />
        <main>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/works" element={<WorksPage />} />
            <Route path="/works/:id" element={<ProjectPage />} />
            <Route path="/writing" element={<WritingPage />} />
            <Route path="/writing/:id" element={<PostPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
};

export default App;