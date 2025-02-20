import { Outlet, Link } from "react-router-dom";
import NavBar from "./NavBar.jsx";

function TeacherDashboard() {
  return (
    <div>
      <NavBar/>
      <div className="dashboard-content">
        <Outlet /> {/* Render nested routes here */}
      </div>
    </div>
  );
}

export default TeacherDashboard;
