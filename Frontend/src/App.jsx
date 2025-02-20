import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import {BrowserRouter,Route, Routes} from "react-router-dom"; 
import Login from './components/Login';
import Register from './components/Register';
import NavBar from './components/NavBar';
import TeacherDashboard from './components/teacherDashboard.jsx';
import TeacherProfile from './components/TeacherProfile.jsx';
import TeacherAttendance from './components/TeacherAttendance.jsx';
import TeacherMarks from './components/TeacherMarks.jsx';
import TeacherLessonPlan from './components/teacherLessonPlan.jsx';
import HMDashboard from './components/HMDashboard.jsx';
import HMProfile from './components/HMProfile.jsx';
import HMManageTeachers from './components/HMManageTeachers.jsx';
import HMManageExams from './components/HMManageExams.jsx';
import HMManageTimeTable from './components/HMManageTimeTable.jsx';


function App() {
  return (
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/navbar" element={<NavBar />} />
        <Route path="/teacher" element={<TeacherDashboard />}>
          <Route index element={<TeacherProfile />} />
          <Route path="attendance" element={<TeacherAttendance />} /> 
          <Route path="marks" element={<TeacherMarks/>}/>
          <Route path="lessonPlan" element={<TeacherLessonPlan/>}/>
        </Route>
        <Route path="/hm" element={<HMDashboard />}>
          <Route index element={<HMProfile />} />
          <Route path="manageTeachers" element={<HMManageTeachers />} /> 
          <Route path="manageExams" element={<HMManageExams/>}/>
          <Route path="manageTimeTable" element={<HMManageTimeTable/>}/>
        </Route>

      </Routes>
    </BrowserRouter>
  );
}

export default App