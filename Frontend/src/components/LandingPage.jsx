import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { 
  BookOpen, 
  Clock, 
  ClipboardList, 
  BarChart2, 
  Bell, 
  Users, 
  Calendar, 
  CheckSquare,
  BookOpenCheck,
  School,
  GraduationCap
} from 'lucide-react';

const LandingPage = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800">
      {/* Header / Navigation Bar */}
      <header className="bg-gray-900 shadow-md py-4 sticky top-0 z-50">
        <div className="container mx-auto px-6 flex justify-between items-center">
          <div className="flex items-center">
            <GraduationCap className="h-8 w-8 text-blue-400 mr-2" />
            <div>
              <h1 className="text-2xl font-bold text-blue-400">Edu-StreamLiners</h1>
              <p className="text-sm text-gray-400">Streamlining Educational Management</p>
            </div>
          </div>
          <nav className="hidden md:block">
            <ul className="flex space-x-8 items-center">
              <li>
                <a href="#features" className="text-gray-300 hover:text-blue-400 transition">
                  Features
                </a>
              </li>
              <li>
                <a href="#benefits" className="text-gray-300 hover:text-blue-400 transition">
                  Benefits
                </a>
              </li>
              <li>
                <a href="#about" className="text-gray-300 hover:text-blue-400 transition">
                  About Us
                </a>
              </li>
              <li>
                <a href="#testimonials" className="text-gray-300 hover:text-blue-400 transition">
                  Testimonials
                </a>
              </li>
              <li>
                <Link to="/register" className="bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-md transition">
                  Register
                </Link>
              </li>
              <li>
                <Link to="/login" className="bg-transparent border border-blue-600 text-blue-600 hover:bg-blue-600 hover:text-white py-2 px-4 rounded-md transition">
                  Login
                </Link>
              </li>
            </ul>
          </nav>
          <button 
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className="md:hidden text-gray-300 focus:outline-none"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16m-7 6h7"></path>
            </svg>
          </button>
        </div>
        {isMenuOpen && (
          <nav className="md:hidden bg-gray-900 px-6 py-4">
            <ul className="flex flex-col space-y-4">
              <li>
                <a href="#features" className="text-gray-300 hover:text-blue-400 transition" onClick={() => setIsMenuOpen(false)}>
                  Features
                </a>
              </li>
              <li>
                <a href="#benefits" className="text-gray-300 hover:text-blue-400 transition" onClick={() => setIsMenuOpen(false)}>
                  Benefits
                </a>
              </li>
              <li>
                <a href="#about" className="text-gray-300 hover:text-blue-400 transition" onClick={() => setIsMenuOpen(false)}>
                  About Us
                </a>
              </li>
              <li>
                <a href="#testimonials" className="text-gray-300 hover:text-blue-400 transition" onClick={() => setIsMenuOpen(false)}>
                  Testimonials
                </a>
              </li>
              <li>
                <Link to="/register" className="bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-md transition" onClick={() => setIsMenuOpen(false)}>
                  Register
                </Link>
              </li>
              <li>
                <Link to="/login" className="bg-transparent border border-blue-600 text-blue-600 hover:bg-blue-600 hover:text-white py-2 px-4 rounded-md transition" onClick={() => setIsMenuOpen(false)}>
                  Login
                </Link>
              </li>
            </ul>
          </nav>
        )}
      </header>

      {/* Hero Section */}
      <section className="py-16 md:py-24">
        <div className="container mx-auto px-6 flex flex-col md:flex-row items-center">
          <div className="md:w-1/2 mb-12 md:mb-0">
            <h1 className="text-4xl md:text-5xl font-bold text-gray-100 leading-tight mb-6">
              Empowering Teachers, Enhancing Education
            </h1>
            <p className="text-lg text-gray-300 mb-8">
              Reduce administrative burden and focus on what matters most – teaching and student engagement.
              Our comprehensive school management system streamlines attendance, performance tracking, and communication.
            </p>
            <div className="flex space-x-4">
              <Link to="/register" className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-6 rounded-md transition">
                Get Started
              </Link>
              <a href="#features" className="bg-transparent border border-gray-500 hover:border-blue-600 text-gray-300 hover:text-blue-400 font-medium py-3 px-6 rounded-md transition">
                Learn More
              </a>
            </div>
          </div>
          <div className="md:w-1/2 flex justify-center">
            <img 
              src="http://bangalorenewsnetwork.com/newspics/2018/august/1caa45e1234260b64bf67be8aaeb2568/1caa4_Halasahalli.jpg" 
              alt="School Management System Dashboard" 
              className="rounded-lg shadow-xl object-cover max-w-full h-auto"
              style={{ maxHeight: '400px', width: 'auto' }}
              onError={(e) => {
                e.target.onerror = null;
                e.target.src = "https://cdn.pixabay.com/photo/2015/07/17/22/43/student-849825_1280.jpg";
              }}
            />
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-16 bg-gray-800">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white mb-4">Comprehensive Features</h2>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Our School Management System automates critical tasks to save time and improve efficiency.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <div className="bg-gray-700 rounded-lg p-8 transition hover:shadow-lg">
              <BarChart2 className="w-12 h-12 text-blue-400 mb-4" />
              <h3 className="text-xl font-semibold text-white mb-3">Student Performance Tracking</h3>
              <p className="text-gray-300">
                Comprehensive view of student progress, including attendance and marks, with shareable reports for parents.
              </p>
            </div>
            
            {/* Feature 2 */}
            <div className="bg-gray-700 rounded-lg p-8 transition hover:shadow-lg">
              <CheckSquare className="w-12 h-12 text-blue-400 mb-4" />
              <h3 className="text-xl font-semibold text-white mb-3">Flexible Attendance Management</h3>
              <p className="text-gray-300">
                Mark attendance online or offline with Excel upload capability for areas with limited connectivity.
              </p>
            </div>
            
            {/* Feature 3 */}
            <div className="bg-gray-700 rounded-lg p-8 transition hover:shadow-lg">
              <Bell className="w-12 h-12 text-blue-400 mb-4" />
              <h3 className="text-xl font-semibold text-white mb-3">Automated Notifications</h3>
              <p className="text-gray-300">
                Instant SMS alerts to parents for student absences, ensuring timely communication and improved safety.
              </p>
            </div>
            
            {/* Feature 4 */}
            <div className="bg-gray-700 rounded-lg p-8 transition hover:shadow-lg">
              <ClipboardList className="w-12 h-12 text-blue-400 mb-4" />
              <h3 className="text-xl font-semibold text-white mb-3">Marks Management</h3>
              <p className="text-gray-300">
                Easily record and track student marks, generate reports, and identify performance trends.
              </p>
            </div>
            
            {/* Feature 5 */}
            <div className="bg-gray-700 rounded-lg p-8 transition hover:shadow-lg">
              <BookOpenCheck className="w-12 h-12 text-blue-400 mb-4" />
              <h3 className="text-xl font-semibold text-white mb-3">Lesson Planning</h3>
              <p className="text-gray-300">
                Plan, track, and evaluate lessons efficiently, improving teaching quality and student outcomes.
              </p>
            </div>
            
            {/* Feature 6 */}
            <div className="bg-gray-700 rounded-lg p-8 transition hover:shadow-lg">
              <Calendar className="w-12 h-12 text-blue-400 mb-4" />
              <h3 className="text-xl font-semibold text-white mb-3">Timetable & Exam Management</h3>
              <p className="text-gray-300">
                Create and manage timetables, schedule exams, and assign substitute teachers with ease.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section id="benefits" className="py-16 bg-gray-900">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white mb-4">Why Choose Edu-StreamLiners?</h2>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Our platform offers significant advantages for schools, teachers, and parents.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
            <div>
              <ul className="space-y-6">
                <li className="flex items-start">
                  <div className="bg-green-900 p-2 rounded-full mr-4">
                    <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-1">Time-Saving Automation</h3>
                    <p className="text-gray-300">Reduces manual workload by 60%, saving up to 10 hours per week for teachers.</p>
                  </div>
                </li>
                <li className="flex items-start">
                  <div className="bg-green-900 p-2 rounded-full mr-4">
                    <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-1">Enhanced Parent Engagement</h3>
                    <p className="text-gray-300">Real-time notifications and performance tracking foster better school-parent communication.</p>
                  </div>
                </li>
                <li className="flex items-start">
                  <div className="bg-green-900 p-2 rounded-full mr-4">
                    <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-1">Works Offline</h3>
                    <p className="text-gray-300">Excel upload feature enables use in areas with limited internet connectivity.</p>
                  </div>
                </li>
              </ul>
            </div>
            <div className="relative">
              <div className="bg-gray-800 p-8 rounded-lg shadow-xl relative z-10">
                <h3 className="text-2xl font-bold text-blue-400 mb-4">Impact Statistics</h3>
                <div className="grid grid-cols-2 gap-6">
                  <div className="text-center">
                    <p className="text-4xl font-bold text-white">60%</p>
                    <p className="text-gray-300">Reduction in Administrative Work</p>
                  </div>
                  <div className="text-center">
                    <p className="text-4xl font-bold text-white">10hrs</p>
                    <p className="text-gray-300">Weekly Time Saved</p>
                  </div>
                  <div className="text-center">
                    <p className="text-4xl font-bold text-white">85%</p>
                    <p className="text-gray-300">Increase in Parent Engagement</p>
                  </div>
                  <div className="text-center">
                    <p className="text-4xl font-bold text-white">100%</p>
                    <p className="text-gray-300">Accurate Resource Allocation</p>
                  </div>
                </div>
              </div>
              <div className="absolute inset-0 bg-blue-900 rounded-lg transform translate-x-4 translate-y-4"></div>
            </div>
          </div>
        </div>
      </section>

      {/* About Us Section - Team Photo Removed */}
      <section id="about" className="py-16 bg-gray-800">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white mb-4">About Edu-StreamLiners</h2>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Driven by firsthand experience and technical expertise to transform education management.
            </p>
          </div>
          <div className="flex flex-col items-center">
            <div className="w-full">
              <h3 className="text-2xl font-semibold text-white mb-6 text-center">Our Mission</h3>
              <p className="text-gray-300 mb-6 text-center">
                At Edu-StreamLiners, we're passionate about freeing teachers from administrative burdens so they can focus on what they do best – inspiring and educating students. With team members who have firsthand experience in the education sector, we understand the challenges educators face every day.
              </p>
              <p className="text-gray-300 mb-6 text-center">
                Our team combines deep educational insights with technical expertise in computer science to create practical solutions that make a real difference in schools across diverse environments.
              </p>
              <div className="bg-blue-50 p-6 rounded-lg border-l-4 border-blue-600 max-w-md mx-auto">
                <h4 className="text-lg font-semibold text-gray-800 mb-2">Our Team</h4>
                <p className="text-gray-700">
                  - Prajwal P Shetti<br />
                  - Shreyas D K<br />
                  - Somanath Mikali<br />
                  - Vishesh P Gowda
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section id="testimonials" className="py-16 bg-gray-900">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white mb-4">What Educators Are Saying</h2>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Feedback from teachers and school administrators using our system.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Testimonial 1 */}
            <div className="bg-gray-700 p-8 rounded-lg shadow-md">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-blue-900 rounded-full flex items-center justify-center mr-4">
                  <Users className="w-6 h-6 text-blue-400" />
                </div>
                <div>
                  <h4 className="text-lg font-semibold text-white">Lakshmi Devi</h4>
                  <p className="text-gray-300">Headmistress, Govt. High School</p>
                </div>
              </div>
              <p className="text-gray-300 italic">
                "The attendance and resource management features have transformed how we run our school. We now have accurate data for mid-day meal allocation, and the time saved on paperwork lets us focus on student needs."
              </p>
              <div className="flex mt-4">
                {[1, 2, 3, 4, 5].map((star) => (
                  <svg key={star} className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"></path>
                  </svg>
                ))}
              </div>
            </div>
            
            {/* Testimonial 2 */}
            <div className="bg-gray-700 p-8 rounded-lg shadow-md">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-blue-900 rounded-full flex items-center justify-center mr-4">
                  <BookOpen className="w-6 h-6 text-blue-400" />
                </div>
                <div>
                  <h4 className="text-lg font-semibold text-white">Ramesh Kumar</h4>
                  <p className="text-gray-300">Science Teacher, Modern School</p>
                </div>
              </div>
              <p className="text-gray-300 italic">
                "The lesson planning feature has made me a better teacher. I can track what works and what doesn't, and the automated attendance system gives me more time to prepare engaging lessons."
              </p>
              <div className="flex mt-4">
                {[1, 2, 3, 4, 5].map((star) => (
                  <svg key={star} className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"></path>
                  </svg>
                ))}
              </div>
            </div>
            
            {/* Testimonial 3 */}
            <div className="bg-gray-700 p-8 rounded-lg shadow-md">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-blue-900 rounded-full flex items-center justify-center mr-4">
                  <School className="w-6 h-6 text-blue-400" />
                </div>
                <div>
                  <h4 className="text-lg font-semibold text-white">Priya Sharma</h4>
                  <p className="text-gray-300">Parent, Class 8 Student</p>
                </div>
              </div>
              <p className="text-gray-300 italic">
                "As a working parent, I appreciate getting SMS updates about my child's attendance. The performance reports help me understand where my child needs support without waiting for parent-teacher meetings."
              </p>
              <div className="flex mt-4">
                {[1, 2, 3, 4, 5].map((star) => (
                  <svg key={star} className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"></path>
                  </svg>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Call to Action Section */}
      <section className="py-20 bg-gradient-to-r from-blue-700 to-blue-900 text-white">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6">Ready to Transform Your School's Management?</h2>
          <p className="text-xl mb-10 max-w-2xl mx-auto">
            Join schools across the country that are saving time, improving communication, and enhancing education quality.
          </p>
          <div className="flex flex-col sm:flex-row justify-center space-y-4 sm:space-y-0 sm:space-x-6">
            <Link to="/register" className="bg-white text-blue-700 hover:bg-gray-100 font-bold py-3 px-8 rounded-md transition">
              Register Now
            </Link>
            <Link to="/login" className="bg-transparent border-2 border-white text-white hover:bg-white hover:text-blue-700 font-bold py-3 px-8 rounded-md transition">
              Login
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <h3 className="text-xl font-semibold mb-4">Edu-StreamLiners</h3>
              <p className="text-gray-400">
                Streamlining educational management to empower teachers and enhance student learning.
              </p>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">Quick Links</h4>
              <ul className="space-y-2">
                <li><a href="#features" className="text-gray-400 hover:text-white transition">Features</a></li>
                <li><a href="#benefits" className="text-gray-400 hover:text-white transition">Benefits</a></li>
                <li><a href="#about" className="text-gray-400 hover:text-white transition">About Us</a></li>
                <li><a href="#testimonials" className="text-gray-400 hover:text-white transition">Testimonials</a></li>
              </ul>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">Support</h4>
              <ul className="space-y-2">
                <li><a href="#" className="text-gray-400 hover:text-white transition">Help Center</a></li>
                <li><a href="#" className="text-gray-400 hover:text-white transition">Documentation</a></li>
                <li><a href="#" className="text-gray-400 hover:text-white transition">Contact Us</a></li>
                <li><a href="#" className="text-gray-400 hover:text-white transition">FAQs</a></li>
              </ul>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">Contact</h4>
              <p className="text-gray-400">Email: info@edustreamliners.com</p>
              <p className="text-gray-400">Phone: +91 9380005468</p>
              <div className="flex space-x-4 mt-4">
                <a href="#" className="text-gray-400 hover:text-white transition">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M22 12c0-5.523-4.477-10-10-10S2 6.477 2 12c0 4.991 3.657 9.128 8.438 9.878v-6.987h-2.54V12h2.54V9.797c0-2.506 1.492-3.89 3.777-3.89 1.094 0 2.238.195 2.238.195v2.46h-1.26c-1.243 0-1.63.771-1.63 1.562V12h2.773l-.443 2.89h-2.33v6.988C18.343 21.128 22 16.991 22 12z"></path>
                  </svg>
                </a>
                <a href="#" className="text-gray-400 hover:text-white transition">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8.29 20.251c7.547 0 11.675-6.253 11.675-11.675 0-.178 0-.355-.012-.53A8.348 8.348 0 0022 5.92a8.19 8.19 0 01-2.357.646 4.118 4.118 0 001.804-2.27 8.224 8.224 0 01-2.605.996 4.107 4.107 0 00-6.993 3.743 11.65 11.65 0 01-8.457-4.287 4.106 4.106 0 001.27 5.477A4.072 4.072 0 012.8 9.713v.052a4.105 4.105 0 003.292 4.022 4.095 4.095 0 01-1.853.07 4.108 4.108 0 003.834 2.85A8.233 8.233 0 012 18.407a11.616 11.616 0 006.29 1.84"></path>
                  </svg>
                </a>
                <a href="#" className="text-gray-400 hover:text-white transition">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd"></path>
                  </svg>
                </a>
                <a href="#" className="text-gray-400 hover:text-white transition">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10c5.51 0 10-4.48 10-10S17.51 2 12 2zm6.605 4.61a8.502 8.502 0 011.93 5.314c-.281-.054-3.101-.629-5.943-.271-.065-.141-.12-.293-.184-.445a25.416 25.416 0 00-.564-1.236c3.145-1.28 4.577-3.124 4.761-3.362zM12 3.475c2.17 0 4.154.813 5.662 2.148-.152.216-1.443 1.941-4.48 3.08-1.399-2.57-2.95-4.675-3.189-5A8.687 8.687 0 0112 3.475zm-3.633.803a53.896 53.896 0 013.167 4.935c-3.992 1.063-7.517 1.04-7.896 1.04a8.581 8.581 0 014.729-5.975zM3.453 12.01v-.26c.37.01 4.512.065 8.775-1.215.25.477.477.965.694 1.453-.109.033-.228.065-.336.098-4.404 1.42-6.747 5.303-6.942 5.629a8.522 8.522 0 01-2.19-5.705zM12 20.547a8.482 8.482 0 01-5.239-1.8c.152-.315 1.888-3.656 6.703-5.337.022-.01.033-.01.054-.022a35.318 35.318 0 011.823 6.475 8.4 8.4 0 01-3.341.684zm4.761-1.465c-.086-.52-.542-3.015-1.659-6.084 2.679-.423 5.022.271 5.314.369a8.468 8.468 0 01-3.655 5.715z"></path>
                  </svg>
                </a>
              </div>
            </div>
          </div>
          <div className="border-t border-gray-700 mt-10 pt-8 text-center">
            <p className="text-gray-400">
              &copy; {new Date().getFullYear()} Edu-StreamLiners. All rights reserved.
            </p>
            <div className="flex justify-center space-x-6 mt-4">
              <a href="#" className="text-gray-400 hover:text-white text-sm transition">Privacy Policy</a>
              <a href="#" className="text-gray-400 hover:text-white text-sm transition">Terms of Service</a>
              <a href="#" className="text-gray-400 hover:text-white text-sm transition">Cookie Policy</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
