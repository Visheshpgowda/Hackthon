import React from 'react'
import NavBar from "./NavBar.jsx";
import HMNavBar from './HMNavBar.jsx';
import { Outlet, Link } from "react-router-dom";

function HMDashboard() {
    return (
        <div>
          <HMNavBar/>
          <div className="dashboard-content">
            <Outlet /> {/* Render nested routes here */}
          </div>
        </div>
    );
}

export default HMDashboard