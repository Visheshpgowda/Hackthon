import { Student } from "../../models/student.models.js";

export const addStudent = async(req, res) => {
    console.log("i am hit here");
    
    try {
        const { name, roll_no, email, phone_no, address, dob, class: studentClass } = req.body;
        
        // Basic null/empty checks
        if (!name || !roll_no || !email || !phone_no || !address || !dob || !studentClass) {
            return res.status(400).json({ message: "All fields are required" });
        }
        
        const student = new Student({
            name,
            roll_no,
            email,
            phone_no,
            address,
            dob,
            class: studentClass
        });
        
        await student.save();
        res.status(201).json({ student });
    } catch (error) {
        // This will catch duplicate key errors from MongoDB
        res.status(409).json({ message: error.message });
    }
}