import React, { useState } from "react";
import { Users, Calendar as CalendarIcon, CheckCircle, XCircle, Loader2 } from "lucide-react";

const TeacherAttendance = () => {
  // ... previous state declarations remain the same ...
  const [selectedClass, setSelectedClass] = useState("");
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [students, setStudents] = useState([]);
  const [attendance, setAttendance] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const [animateHeader, setAnimateHeader] = useState(true);

  // ... previous functions remain the same ...
  const classes = [
    { id: "1", name: "10A" },
    { id: "2", name: "10B" },
    { id: "3", name: "10C" },
    { id: "4", name: "9A" },
    { id: "5", name: "9B" },
    { id: "6", name: "9C" },
  ];

  const fetchStudents = async () => {
    if (!selectedClass) return;
    setIsLoading(true);
    try {
      const response = await fetch("http://localhost:8080/api/v1/students/getstudents", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ class: selectedClass })
      });
      const data = await response.json();
      setStudents([]);
      setTimeout(() => {
        setStudents(data.studentList || []);
        const initialAttendance = data.studentList.reduce((acc, student) => ({
          ...acc,
          [student._id]: false
        }), {});
        setAttendance(initialAttendance);
      }, 300);
    } catch (error) {
      console.error("Error fetching students:", error);
    }
    setIsLoading(false);
  };

  const handleAttendanceChange = (studentId) => {
    setAttendance(prev => ({ ...prev, [studentId]: !prev[studentId] }));
  };

  const markAll = (status) => {
    const updatedAttendance = students.reduce((acc, student) => ({
      ...acc,
      [student._id]: status
    }), {});
    setAttendance(updatedAttendance);
  };

  const submitAttendance = async () => {
    setIsSubmitting(true);
    try {
      const formattedDate = selectedDate.toISOString().split('T')[0];
      const attendanceData = {
        classId: selectedClass,
        attendance: Object.entries(attendance).map(([studentId, isPresent]) => ({
          studentId,
          date: formattedDate,
          isPresent
        }))
      };

      await fetch("http://localhost:8080/api/v1/students/addattendance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(attendanceData)
      });

      setShowSuccess(true);
      setTimeout(() => setShowSuccess(false), 3000);
    } catch (error) {
      console.error("Error submitting attendance:", error);
    }
    setIsSubmitting(false);
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6 opacity-0 animate-[fadeIn_0.5s_ease-in-out_forwards]">
      <div className="bg-white rounded-lg shadow-lg overflow-hidden transition-all duration-500 hover:shadow-xl transform hover:-translate-y-1">
        <div className={`p-6 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-indigo-50 transition-all duration-500 ${animateHeader ? 'animate-[slideDown_0.5s_ease-in-out]' : ''}`}>
          <div className="flex items-center space-x-2">
            <Users className="w-6 h-6 text-blue-500 animate-[spin_20s_linear_infinite]" />
            <h2 className="text-2xl font-bold text-gray-800 animate-[fadeIn_1s_ease-in-out]">Teacher Attendance</h2>
          </div>
        </div>
        
        <div className="p-6 space-y-6">
          <div className="flex flex-col sm:flex-row gap-4 animate-[slideIn_0.5s_ease-in-out]">
            <select 
              className="flex h-10 w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-300 transform hover:scale-[1.01]"
              onChange={(e) => setSelectedClass(e.target.value)}
              value={selectedClass}
            >
              <option value="">Select Class</option>
              {classes.map((cls) => (
                <option key={cls.id} value={cls.name}>{cls.name}</option>
              ))}
            </select>
            
            <button 
              onClick={fetchStudents}
              disabled={!selectedClass || isLoading}
              className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-all duration-300 disabled:bg-gray-300 disabled:cursor-not-allowed min-w-[120px] transform hover:scale-105 active:scale-95 flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Loading...
                </>
              ) : "Get Students"}
            </button>
          </div>

          <div className="flex items-center gap-4 animate-[slideIn_0.7s_ease-in-out]">
            <CalendarIcon className="w-5 h-5 text-gray-500" />
            <input
              type="date"
              value={selectedDate.toISOString().split('T')[0]}
              onChange={(e) => setSelectedDate(new Date(e.target.value))}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-300 transform hover:scale-[1.01]"
            />
          </div>

          {students.length > 0 && (
            <div className="space-y-4 animate-[fadeIn_0.5s_ease-in-out]">
              <div className="flex gap-2 justify-end">
                <button
                  onClick={() => markAll(true)}
                  className="px-4 py-2 bg-green-50 text-green-600 rounded-md hover:bg-green-100 transition-all duration-300 flex items-center transform hover:scale-105 active:scale-95"
                >
                  <CheckCircle className="w-4 h-4 mr-2" />
                  Mark All Present
                </button>
                <button
                  onClick={() => markAll(false)}
                  className="px-4 py-2 bg-red-50 text-red-600 rounded-md hover:bg-red-100 transition-all duration-300 flex items-center transform hover:scale-105 active:scale-95"
                >
                  <XCircle className="w-4 h-4 mr-2" />
                  Mark All Absent
                </button>
              </div>

              <div className="space-y-2">
                {students.map((student, index) => (
                  <div
                    key={student._id}
                    className="flex items-center justify-between p-4 rounded-lg bg-gray-50 hover:bg-gray-100 transition-all duration-300 transform hover:-translate-x-1 hover:shadow-md"
                    style={{
                      animation: `slideIn 0.5s ease-in-out forwards`,
                      animationDelay: `${index * 0.1}s`,
                      opacity: 0
                    }}
                  >
                    <div>
                      <p className="font-medium text-gray-900">{student.name}</p>
                      <p className="text-sm text-gray-500">Roll No: {student.roll_no}</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer group">
                      <input
                        type="checkbox"
                        checked={attendance[student._id] || false}
                        onChange={() => handleAttendanceChange(student._id)}
                        className="sr-only peer"
                      />
                      <div className="w-11 h-6 bg-gray-200 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600 group-hover:ring-2 group-hover:ring-blue-300"></div>
                      <span className="ml-3 text-sm font-medium text-gray-900 transition-all duration-300">
                        {attendance[student._id] ? 'Present' : 'Absent'}
                      </span>
                    </label>
                  </div>
                ))}
              </div>

              <button
                onClick={submitAttendance}
                disabled={isSubmitting}
                className="w-full px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-all duration-300 disabled:bg-gray-300 disabled:cursor-not-allowed transform hover:scale-[1.02] active:scale-98 flex items-center justify-center gap-2"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Submitting...
                  </>
                ) : "Submit Attendance"}
              </button>
            </div>
          )}
        </div>
      </div>

      {showSuccess && (
        <div className="fixed bottom-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg animate-[slideInRight_0.5s_ease-in-out] flex items-center gap-2">
          <CheckCircle className="w-5 h-5" />
          Attendance submitted successfully!
        </div>
      )}

      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateX(-20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
        
        @keyframes slideDown {
          from {
            opacity: 0;
            transform: translateY(-20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes slideInRight {
          from {
            opacity: 0;
            transform: translateX(100px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
      `}</style>
    </div>
  );
};

export default TeacherAttendance;