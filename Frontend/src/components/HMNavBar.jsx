import React from 'react'
import { NavLink } from "react-router-dom";
import { Search, ShoppingCart, Package } from 'lucide-react';

function HMNavBar() {
  return (
    <nav className="top-0 z-50 w-full backdrop-blur-sm bg-pink-200 border-b border-gray-200 p-4">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        {/* Left side - Menu Items */}
        <div className="flex">
          <NavLink to="/hm" className="flex items-center gap-2 text-lg font-medium text-gray-700 hover:text-blue-600 hover:bg-white/50 p-2 rounded-lg">
            Profile
          </NavLink>

          <NavLink to="/hm/manageTeachers" className="flex items-center gap-2 text-lg font-medium text-gray-700 hover:text-blue-600 hover:bg-white/50 p-2 rounded-lg">
          manageTeachers
          </NavLink>

          <NavLink to="/hm/manageExams" className="flex items-center gap-2 text-lg font-medium text-gray-700 hover:text-blue-600 hover:bg-white/50 p-2 rounded-lg">
          manageExams
          </NavLink>

          <NavLink to="/hm/manageTimeTable" className="flex items-center gap-2 text-lg font-medium text-gray-700 hover:text-blue-600 hover:bg-white/50 p-2 rounded-lg">
          manageTimeTable
          </NavLink>
        </div>
      </div>
    </nav>
  )
}

export default HMNavBar