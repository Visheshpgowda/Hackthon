import { Student } from "../../models/student.models";

export const getStudentListByClass = async (req, res) => {
    try {
        const { class: className } = req.body;  // Rename class to className
        if (!className) {
            return res.status(400).json({ message: "Class field is required" });
        }
        
        const studentList = await Student.find({ class: className });
        res.status(200).json({ studentList });
    } catch (error) {
        res.status(500).json({ message: "Internal Server Error", error: error.message });
    }
};
