import React, { useState } from "react";
import { 
  Users, 
  Calendar as CalendarIcon, 
  CheckCircle, 
  XCircle, 
  Loader2, 
  Upload,
  Download,
  Globe,
  FileSpreadsheet
} from "lucide-react";
import * as XLSX from 'xlsx';

const TeacherAttendance = () => {
  const [selectedClass, setSelectedClass] = useState("");
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [students, setStudents] = useState([]);
  const [attendance, setAttendance] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const [animateHeader, setAnimateHeader] = useState(true);
  const [mode, setMode] = useState(""); // "live" or "excel"

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
      setStudents(data.studentList || []);
      const initialAttendance = data.studentList.reduce((acc, student) => ({
        ...acc,
        [student._id]: false
      }), {});
      setAttendance(initialAttendance);
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

  const downloadTemplate = () => {
    const template = students.map(student => ({
      'Roll No': student.roll_no,
      'Name': student.name,
      'Attendance': '',  // Empty column for teachers to fill
      'Date': selectedDate.toISOString().split('T')[0]
    }));

    const ws = XLSX.utils.json_to_sheet(template);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Attendance");
    
    // Save the file
    XLSX.writeFile(wb, `attendance_template_${selectedClass}_${selectedDate.toISOString().split('T')[0]}.xlsx`);
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = async (e) => {
      const data = new Uint8Array(e.target.result);
      const workbook = XLSX.read(data, { type: 'array' });
      const sheetName = workbook.SheetNames[0];
      const worksheet = workbook.Sheets[sheetName];
      const jsonData = XLSX.utils.sheet_to_json(worksheet);

      // Convert Excel data to attendance format
      const attendanceData = jsonData.reduce((acc, row) => {
        const student = students.find(s => s.roll_no === row['Roll No']);
        if (student) {
          acc[student._id] = row['Attendance'].toLowerCase() === 'present';
        }
        return acc;
      }, {});

      setAttendance(attendanceData);
      await submitAttendance(attendanceData);
    };

    reader.readAsArrayBuffer(file);
  };

  const submitAttendance = async (attendanceData = attendance) => {
    setIsSubmitting(true);
    try {
      const formattedDate = selectedDate.toISOString().split('T')[0];
      const data = {
        classId: selectedClass,
        attendance: Object.entries(attendanceData).map(([studentId, isPresent]) => ({
          studentId,
          date: formattedDate,
          isPresent
        }))
      };

      await fetch("http://localhost:8080/api/v1/students/addattendance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
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
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className={`p-6 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-indigo-50`}>
          <div className="flex items-center space-x-2">
            <Users className="w-6 h-6 text-blue-500" />
            <h2 className="text-2xl font-bold text-gray-800">Enter Student Attendance</h2>
          </div>
        </div>
        
        <div className="p-6 space-y-6">
          {/* Class Selection */}
          <div className="flex flex-col sm:flex-row gap-4">
            <select 
              className="flex h-10 w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
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
              className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-all duration-300 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Loading...
                </>
              ) : "Get Students"}
            </button>
          </div>

          {/* Mode Selection */}
          {selectedClass && students.length > 0 && !mode && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <button
                onClick={() => setMode("live")}
                className="p-6 border rounded-lg hover:border-blue-500 transition-all duration-300 flex flex-col items-center gap-3"
              >
                <Globe className="w-8 h-8 text-blue-500" />
                <span className="font-medium">Mark Live Attendance</span>
              </button>
              <button
                onClick={() => setMode("excel")}
                className="p-6 border rounded-lg hover:border-blue-500 transition-all duration-300 flex flex-col items-center gap-3"
              >
                <FileSpreadsheet className="w-8 h-8 text-green-500" />
                <span className="font-medium">Upload Excel File</span>
              </button>
            </div>
          )}

          {/* Date Selection */}
          {mode && (
            <div className="flex items-center gap-4">
              <CalendarIcon className="w-5 h-5 text-gray-500" />
              <input
                type="date"
                value={selectedDate.toISOString().split('T')[0]}
                onChange={(e) => setSelectedDate(new Date(e.target.value))}
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          )}

          {/* Excel Mode */}
          {mode === "excel" && (
            <div className="space-y-4">
              <button
                onClick={downloadTemplate}
                className="w-full px-4 py-3 bg-green-500 text-white rounded-md hover:bg-green-600 transition-all duration-300 flex items-center justify-center gap-2"
              >
                <Download className="w-5 h-5" />
                Download Template
              </button>

              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                <input
                  type="file"
                  accept=".xlsx,.xls"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer flex flex-col items-center gap-2"
                >
                  <Upload className="w-8 h-8 text-gray-400" />
                  <span className="text-sm text-gray-600">Click to upload attendance file</span>
                </label>
              </div>
            </div>
          )}

          {/* Live Attendance Mode */}
          {mode === "live" && students.length > 0 && (
            <div className="space-y-4">
              <div className="flex gap-2 justify-end">
                <button
                  onClick={() => markAll(true)}
                  className="px-4 py-2 bg-green-50 text-green-600 rounded-md hover:bg-green-100 transition-all duration-300 flex items-center"
                >
                  <CheckCircle className="w-4 h-4 mr-2" />
                  Mark All Present
                </button>
                <button
                  onClick={() => markAll(false)}
                  className="px-4 py-2 bg-red-50 text-red-600 rounded-md hover:bg-red-100 transition-all duration-300 flex items-center"
                >
                  <XCircle className="w-4 h-4 mr-2" />
                  Mark All Absent
                </button>
              </div>

              <div className="space-y-2">
                {students.map((student) => (
                  <div
                    key={student._id}
                    className="flex items-center justify-between p-4 rounded-lg bg-gray-50 hover:bg-gray-100 transition-all duration-300"
                  >
                    <div>
                      <p className="font-medium text-gray-900">{student.name}</p>
                      <p className="text-sm text-gray-500">Roll No: {student.roll_no}</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={attendance[student._id] || false}
                        onChange={() => handleAttendanceChange(student._id)}
                        className="sr-only peer"
                      />
                      <div className="w-11 h-6 bg-gray-200 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                      <span className="ml-3 text-sm font-medium text-gray-900">
                        {attendance[student._id] ? 'Present' : 'Absent'}
                      </span>
                    </label>
                  </div>
                ))}
              </div>

              <button
                onClick={() => submitAttendance()}
                disabled={isSubmitting}
                className="w-full px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-all duration-300 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center gap-2"
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
        <div className="fixed bottom-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg flex items-center gap-2">
          <CheckCircle className="w-5 h-5" />
          Attendance submitted successfully!
        </div>
      )}
    </div>
  );
};

export default TeacherAttendance;