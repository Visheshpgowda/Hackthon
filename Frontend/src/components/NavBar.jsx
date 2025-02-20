import { NavLink } from "react-router-dom";
import { Search, ShoppingCart, Package } from 'lucide-react';

function NavBar() {
  return (
    <nav className="top-0 z-50 w-full backdrop-blur-sm bg-pink-200 border-b border-gray-200 p-4">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        {/* Left side - Menu Items */}
        <div className="flex">
          <NavLink to="/teacher" className="flex items-center gap-2 text-lg font-medium text-gray-700 hover:text-blue-600 hover:bg-white/50 p-2 rounded-lg">
            Profile
          </NavLink>

          <NavLink to="/teacher/attendance" className="flex items-center gap-2 text-lg font-medium text-gray-700 hover:text-blue-600 hover:bg-white/50 p-2 rounded-lg">
            Attendance
          </NavLink>

          <NavLink to="/teacher/marks" className="flex items-center gap-2 text-lg font-medium text-gray-700 hover:text-blue-600 hover:bg-white/50 p-2 rounded-lg">
            Marks
          </NavLink>

          <NavLink to="/teacher/lessonPlan" className="flex items-center gap-2 text-lg font-medium text-gray-700 hover:text-blue-600 hover:bg-white/50 p-2 rounded-lg">
            Lesson Plan
          </NavLink>
        </div>
      </div>
    </nav>
  );
}

export default NavBar;
