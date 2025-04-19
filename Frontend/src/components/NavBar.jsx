import React, { useState, useEffect } from 'react';
import { NavLink, useNavigate } from "react-router-dom";
import { 
  UserCircle, ClipboardList, BookOpen, LineChart,
  GraduationCap, LogOut, Menu, X, AlertCircle,
  CheckCircle, Loader2, Speaker,
  Bot
} from 'lucide-react';

const Toast = ({ message, type, onClose }) => (
  <div className={`fixed top-4 right-4 ${type === 'success' ? 'bg-green-500' : 'bg-red-500'} 
    text-white px-4 py-2 rounded shadow-lg flex items-center gap-2 z-50 animate-slideIn`}>
    {type === 'success' ? <CheckCircle className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
    <span>{message}</span>
    <button onClick={onClose} className="hover:opacity-80"><X className="w-4 h-4" /></button>
  </div>
);

const NavBar = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [toast, setToast] = useState(null);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const navigate = useNavigate();

  const navItems = [
    { path: '/teacher', icon: UserCircle, label: 'Profile' },
    { path: '/teacher/attendance', icon: ClipboardList, label: 'Fill Attendance' },
    { path: '/teacher/viewClassAttendance', icon: ClipboardList, label: 'View Attendance' },
    { path: '/teacher/marks', icon: LineChart, label: 'Marks' },
    { path: '/teacher/lessonPlan', icon: BookOpen, label: 'Plan' },
    { path: '/teacher/studentPerformance', icon: GraduationCap, label: 'Student' },
    { path: '/teacher/announcements', icon: Speaker, label: 'Announcements' },
    { path: '/teacher/chatbot', icon: Bot, label: 'Chatbot'}
  ];

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_BASE_URL}/verify/verifyLogin`, {
        credentials: 'include'
      });
      setIsAuthenticated(response.ok);
      if (!response.ok) setToast({ message: 'Please log in', type: 'error' });
    } catch (error) {
      console.error('Auth check failed:', error);
      setIsAuthenticated(false);
      setToast({ message: 'Connection error', type: 'error' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_BASE_URL}/users/logout`, {
        method: 'POST',
        credentials: 'include'
      });
      if (response.ok) {
        setToast({ message: 'Logged out', type: 'success' });
        setTimeout(() => navigate('/'), 0.6);
      }
    } catch (error) {
      setToast({ message: 'Logout failed', type: 'error' });
    }
  };

  if (isLoading) {
    return (
      <div className="w-full h-16 bg-gray-900 flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin text-blue-400" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="w-full min-h-screen bg-gray-900 flex items-center justify-center p-4">
        <div className="bg-gray-800 p-6 rounded-xl shadow-lg max-w-sm w-full text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-white mb-3">Access Denied</h2>
          <button
            onClick={() => navigate('/')}
            className="w-full bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded-lg"
          >
            Login
          </button>
        </div>
      </div>
    );
  }

  return (
    <>
      {toast && <Toast {...toast} onClose={() => setToast(null)} />}
      
      <nav className="sticky top-0 z-40 w-full bg-gray-900 shadow-lg border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex h-16 items-center justify-between">
            {/* Mobile menu button */}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="lg:hidden text-gray-300 hover:text-white"
            >
              {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>

            {/* Desktop navigation */}
            <div className="hidden lg:flex items-center space-x-4">
              {navItems.map(({ path, icon: Icon, label }) => (
                <NavLink
                  key={path}
                  to={path}
                  end={path === '/teacher'}
                  className={({ isActive }) =>
                    `flex items-center gap-2 px-3 py-2 rounded-lg transition-all ${
                      isActive ? 'text-blue-400 bg-gray-800' : 'text-gray-300 hover:text-blue-400 hover:bg-gray-800'
                    }`
                  }
                >
                  <Icon className="w-5 h-5" />
                  <span>{label}</span>
                </NavLink>
              ))}
            </div>

            {/* Right side content */}
            <div className="flex items-center gap-4">
              <span className="text-xl font-bold text-blue-400">Teacher</span>
              <button
                onClick={handleLogout}
                className="text-gray-300 hover:text-red-400 flex items-center gap-2"
              >
                <LogOut className="w-5 h-5" />
                <span className="hidden sm:inline">Logout</span>
              </button>
            </div>
          </div>

          {/* Mobile menu */}
          {isMobileMenuOpen && (
            <div className="lg:hidden pb-4">
              {navItems.map(({ path, icon: Icon, label }) => (
                <NavLink
                  key={path}
                  to={path}
                  end={path === '/teacher'}
                  onClick={() => setIsMobileMenuOpen(false)}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-4 py-3 rounded-lg mb-1 ${
                      isActive ? 'text-blue-400 bg-gray-800' : 'text-gray-300 hover:text-blue-400 hover:bg-gray-800'
                    }`
                  }
                >
                  <Icon className="w-5 h-5" />
                  <span>{label}</span>
                </NavLink>
              ))}
            </div>
          )}
        </div>
      </nav>
    </>
  );
};

export default NavBar;