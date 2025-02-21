import { Student } from "../../models/student.models.js";
import  studentEnrolledSubjects  from "../../models/studentEnrolledSubjects.models.js";

export const getStudentListByClass = async (req, res) => {
    console.log("came to getStudentListByClass");
    
    try {
        const { class: className } = req.body;  // Rename class to className
        if (!className) {
            return res.status(400).json({ message: "Class field is required" });
        }
        
        const studentList = await Student.find({ class: className }); 
        console.log(studentList);
        
        res.status(200).json({ studentList }); 
        
    } catch (error) {
        res.status(500).json({ message: "Internal Server Error", error: error.message });
    }
};

export const getEnrolledSubjects = async (req, res) => {
    console.log("came to enrolled subjects");
    
    try {
        const { studentId } = req.params;  
        const enrolled = await studentEnrolledSubjects.find({ Student_name: studentId });  
        
        if (!enrolled || enrolled.length === 0) {
            return res.status(404).json({ message: "No subjects found for this student" });
        }

        // Take the first entry's Subject_name array
        console.log(enrolled);
        
        const subjects = enrolled[0].Subject_name;
        console.log("Sending subjects for student:", subjects[0]);
        
        res.status(200).json({ subjects });
    } catch (error) {
        res.status(500).json({ message: "Internal Server Error", error: error.message });
    }
};