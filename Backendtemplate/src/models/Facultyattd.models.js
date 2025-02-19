import mongoose, { Schema } from "mongoose";

const facultyAttendanceSchema = new Schema({
    facultyId: {
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    date: {
        type: Date,
        required: true
    },
    status: {
        type: String,
        enum: ['present', 'absent'],
        required: true
    }
}, {
    timestamps: true
});


export const FacultyAttendance = mongoose.model("FacultyAttendance", facultyAttendanceSchema)