import mongoose,{Schema} from 'mongoose';

const studentAttendanceSchema = new 
Schema({
    studentId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Student',
        required: true
    },
    date: {
        type: Date,
        required: true
    },
    status: {
        type: String,
        enum: ['true', 'false'],
        required: true
    }
}, {
    timestamps: true
});

export const StudentAttendance = mongoose.model('StudentAttendance', studentAttendanceSchema);

