import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import {BrowserRouter,Route, Routes} from "react-router-dom"; 
import Login from './components/Login';
import Register from './components/Register';
import NavBar from './components/NavBar';
import TeacherDashboard from './components/TeacherDashboard.jsx';
import TeacherProfile from './components/TeacherProfile.jsx';
import TeacherAttendance from './components/TeacherAttendance.jsx';
import TeacherMarks from './components/TeacherMarks.jsx';
import HMDashboard from './components/HMDashboard.jsx';
import HMProfile from './components/HMProfile.jsx';
import HMManageTeachers from './components/HMManageTeachers.jsx';
import ManageExamsAndInvigilators from './components/HMManageExams.jsx';
import HMManageTimeTable from './components/HMManageTimeTable.jsx';
import StudentListManager from './components/StudentListManager.jsx';
import TeacherLessonPlan from './components/TeacherLessonPlan.jsx';
import TeacherUpcomingExams from './components/TeacherUpcomingExams.jsx'
import ViewAttendance from './components/ViewAttendance.jsx'
import AttendanceReport from './components/AttendanceReport.jsx';
import Announcements from './components/TeacherAnnouncement.jsx';
import LandingPage from './components/LandingPage.jsx'
import TeacherBot from './components/TeacherBot.jsx';

function App() {
  return (
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes>
      <Route path="/" element={<LandingPage />} /> 
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/navbar" element={<NavBar />} />
        <Route path="/teacher" element={<TeacherDashboard />}>
          <Route index element={<TeacherProfile />} />
          <Route path="attendance" element={<TeacherAttendance />} /> 
          <Route path="marks" element={<TeacherMarks/>}/>
          <Route path="viewClassAttendance" element={<ViewAttendance/>}/> 
          <Route path="lessonPlan" element={<TeacherLessonPlan/>}/>
          <Route path='studentPerformance' element={<StudentListManager/>}/>
          <Route path='upcomingExams' element={<TeacherUpcomingExams/>}/>
          <Route path='chatbot' element={<TeacherBot/>}/>
          <Route path='Announcements' element={<Announcements/>}/>
        </Route>
        <Route path="/hm" element={<HMDashboard />}>
          <Route index element={<HMProfile />} />
          <Route path="manageTeachers" element={<HMManageTeachers />} /> 
          <Route path="manageExams" element={<ManageExamsAndInvigilators/>}/>
          <Route path="manageTimeTable" element={<HMManageTimeTable/>}/>
          <Route path="todaysAttendace" element={<AttendanceReport/>}/> 
        </Route>

      </Routes>
    </BrowserRouter>
  );
}

export default App